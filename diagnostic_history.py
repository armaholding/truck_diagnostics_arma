# diagnostic_history.py
"""Temporal consensus engine for truck component diagnostics with yolo confidence tie breaker."""

import logging
from datetime import datetime
import re
import numpy as np
from collections import deque
import cv2
from config import (DIAGNOSTIC_THRESHOLD, EXPECTED_COMPONENT_COUNTS, FRONT_EXPECTED_COMPONENTS, BACK_EXPECTED_COMPONENTS, 
                    WIPER_FRAMES_TO_COLLECT, WIPER_COLLECTION_INTERVAL_SECONDS, WIPER_MIN_FRAMES_FOR_ANALYSIS, WIPER_MOVEMENT_THRESHOLD_RELATIVE,
                    LIGHT_FRAMES_TO_COLLECT, LIGHT_COLLECTION_INTERVAL_SECONDS, LIGHT_GRID_ROWS, LIGHT_GRID_COLS,
                    LIGHT_BRIGHTNESS_CHANGE_THRESHOLD, LIGHT_MIN_FRAMES_FOR_ANALYSIS, LIGHT_MIN_SECTION_GRIDS)

# Configure module-specific logger
logger = logging.getLogger(__name__)

class DiagnosticHistory:
    """
    Track diagnostic history and compute temporal consensus decisions with ignore period.
    
    Implements two-stage consensus voting:
    1. Majority vote on component states/plate numbers across time window
    2. Confidence-based tiebreaker for ambiguous cases
    
    Maintains consensus window wit ignore period to exclude unstable initial frames.
    """
    
    def __init__(self, count_to_decide_consensus=7, ignore_period_seconds=1):
        """
        Initialize diagnostic history tracker with temporal window configuration.
        
        Args:
            count_to_decide_consensus (int): Maximum diagnostic samples to collect (default: 7)
            ignore_period_seconds (int): Duration in seconds to ignore at start of processing for stabilization (default: 1)
        """
        self.count_to_decide_consensus = count_to_decide_consensus # Maximum number of diagnostic samples to collect before stopping
        self.ignore_period = ignore_period_seconds # Initial ignore period to allow for stabilization (e.g., 1 second)
        self.diagnostic_history = []      # List of (timestamp, diagnostics_dict)
        self.component_history = {}       # Track each component's states over time
        self.processing_start_time = None # Timestamp when processing starts (set externally before adding diagnostics)

    def set_start_time(self, start_time):
        """
        Set processing start time for ignore period calculation.
        
        Required before adding diagnostics to properly exclude initial unstable frames.
        
        Args:
            start_time (float): Start timestamp in seconds (from time.time())
        """
        self.processing_start_time = start_time

    def _get_expected_components(self, truck_face):
        """
        Get list of expected component names based on detected truck face.
        
        Args:
            truck_face (str): Detected truck face ("truck_front", "truck_back", or "unknown")
        
        Returns:
            list: List of expected component names for the given truck face
        """
        if truck_face == "truck_front":
            return list(FRONT_EXPECTED_COMPONENTS)
        elif truck_face == "truck_back":
            return list(BACK_EXPECTED_COMPONENTS)
        else:
            return list(EXPECTED_COMPONENT_COUNTS.keys())

    def _get_display_name(self, component):
        """
        Convert internal component name to human-readable singular display name.
        
        Args:
            component (str): Internal component name (e.g., 'mirror', 'plate_number')
        
        Returns:
            str: Singular human-readable display name (e.g., 'Mirror', 'Plate')
        """
        display_map = {
            'mirror': 'Mirror',
            'light_front': 'Front Light',
            'wiper': 'Wiper',
            'mirror_top': 'Top Mirror',
            'plate_number': 'Plate',  # Singular for state consensus
            'light_back': 'Back Light',
            'stand': 'Stand',
            'carrier': 'Carrier',
            'lift': 'Lift'
        }
        return display_map.get(component, component.replace('_', ' ').title())
     
    def _get_component_state(self, component, data):
        """
        Determine component state (✅/⚠️/❌) based on detection count and confidence thresholds.
        
        Args:
            component (str): Component name (e.g., "mirror", "plate_number")
            data (dict): Component detection data with "count", "confidence", and optionally "wiper_moving"/"light_working"
        
        Returns:
            str: State emoji ("✅" = pass, "⚠️" = detected but unreadable, "❌" = fail)
        """
        count = data.get("count", 0)
        confidences = data.get("confidence", [])

        # For wipers, check wiper_moving field first
        if component in ("left_wiper", "right_wiper"):
            wiper_moving = data.get("wiper_moving")
            if wiper_moving is False:
                return "❌"  # Movement analysis failed → state is ❌
            elif wiper_moving is True:
                return "✅"  # Movement analysis passed → state is ✅
            # If None, fall through to YOLO confidence-based logic

        # For front lights, check light_working field first
        if component in ("left_light_front", "right_light_front"):
            light_working = data.get("light_working")
            if light_working is False:
                return "❌"  # Brightness analysis failed → state is ❌
            elif light_working is True:
                return "✅"  # Brightness analysis passed → state is ✅
            # If None, fall through to YOLO confidence-based logic

        # For plate number, check OCR result first
        if component == "plate_number":
            # Plate number state depends on OCR success
            if "number" in data:
                return "✅"
            elif count >= EXPECTED_COMPONENT_COUNTS.get(component, 1) and all(c >= DIAGNOSTIC_THRESHOLD for c in confidences):
                return "⚠️"
            else:
                return "❌"
        
        # For other components (e.g., mirrors, stands), use count + confidence logic
        expected_count = EXPECTED_COMPONENT_COUNTS.get(component, 1)
        if count < expected_count or any(c < DIAGNOSTIC_THRESHOLD for c in confidences):
            return "❌"
        else:
            return "✅"
            
    def _normalize_plate_number(self, plate_str):
        """
        Normalize plate number PRESERVING Arabic characters (U+0600-U+06FF).
        
        Critical: Moroccan plates require Arabic character between digits (e.g., '8343ب1').
        This function keeps digits + Arabic script ONLY - strips everything else.
        
        Args:
            plate_str (str | None): Raw plate number string from OCR
        
        Returns:
            str | None: Normalized string with digits + Arabic characters, or None if invalid
        """
        if not plate_str or not isinstance(plate_str, str):
            return None
        # Keep ONLY digits (0-9) + Arabic script (U+0600-U+06FF)
        normalized = re.sub(r'[^\d\u0600-\u06FF]', '', plate_str)
        return normalized if normalized else None

    def _resolve_consensus_two_stage(self, value_occurrences, value_confidences):
        """
        Resolve consensus using two-stage voting: majority count → confidence tiebreaker.
        
        Stage 1: Select value(s) with highest occurrence count (simple majority)
        Stage 2: If tie (multiple values share max count), use average confidence as tiebreaker
        
        Args:
            value_occurrences (dict): Mapping {value: occurrence_count}
            value_confidences (dict): Mapping {value: [list of confidence scores]}
        
        Returns:
            any | None: Consensus value with highest occurrence (or highest avg confidence if tied), or None if no data
        """
        if not value_occurrences:
            return None

        # Stage 1: Find max occurrence count (simple majority = most frequent value)
        max_count = max(value_occurrences.values())
        
        # Find all values with max count (candidates for majority/tie)
        candidates = [val for val, cnt in value_occurrences.items() if cnt == max_count]
        
        # Single candidate = clear majority winner
        if len(candidates) == 1:
            return candidates[0]
        
        # Stage 2: Tiebreaker - compute average confidence for each candidate
        candidate_avg_confs = {}
        for candidate in candidates:
            confs = value_confidences.get(candidate, [])
            if confs:
                candidate_avg_confs[candidate] = sum(confs) / len(confs)
            else:
                candidate_avg_confs[candidate] = 0.0
        
        # Select candidate with highest average confidence
        best_candidate = max(candidate_avg_confs, key=candidate_avg_confs.get)
        return best_candidate
    
    def _build_consensus_message(self, component, state):
        """
        Build human-readable diagnostic message for component consensus state.
        
        Args:
            component (str): Component name (e.g., "mirror", "plate_number")
            state (str): Consensus state ("✅", "⚠️", "❌", or "❓")
        
        Returns:
            str: Formatted diagnostic message with emoji and component description
        """
        component_names = {
            'mirror': 'mirrors',
            'light_front': 'front lights', 
            'wiper': 'wipers',
            'mirror_top': 'top mirror',
            'plate_number': 'plate number',
            'carrier': 'carrier',
            'lift': 'lift',
            'light_back': 'back lights',
            'stand': 'stands'
        }
        
        display_name = component_names.get(component, component)        
        if state == "✅":
            if component == "plate_number":
                return f"✅ {display_name}: visible with number"
            else:
                return f"✅ {display_name}: ok"
        elif state == "⚠️":
            return f"⚠️ {display_name}: visible but could not be read"
        else:  # "❌"
            if component == "plate_number":
                return f"❌ {display_name}: missing or obscured"
            else:
                return f"❌ {display_name}: missing/broken"
    
    def _build_diagnostics_from_result(self, diagnostics_result):
        """
        Extract diagnostic components from single-frame result for fallback scenarios.
        
        Args:
            diagnostics_result (dict): Single-frame diagnostic result dictionary
        
        Returns:
            tuple: (diagnostics_log, truck_face, enhanced_components)
                - diagnostics_log: List of diagnostic messages
                - truck_face: Detected truck face ("truck_front" or "truck_back")
                - enhanced_components: Component data dictionary
        """
        truck_face = diagnostics_result["truck_face"]
        enhanced_components = diagnostics_result["truck_components"]
        diagnostics_log = diagnostics_result["diagnostics"]
        
        return diagnostics_log, truck_face, enhanced_components

    def _has_arabic(self, text):
        """Check if text contains Arabic script characters (U+0600-U+06FF)."""
        if not text or not isinstance(text, str):
            return False
        return bool(re.search(r'[\u0600-\u06FF]', text))

    def _get_best_detections_in_window(self, component, window_start, window_end):
        """
        Retrieve highest-confidence detections for component within specified time window.
        
        Filters diagnostics to ✅ states only, then returns top N confidences (N = expected count).
        
        Args:
            component (str): Component name to query
            window_start (float): Start timestamp of query window
            window_end (float): End timestamp of query window
        
        Returns:
            list: Sorted list of top confidence values (descending order), limited to expected count
        """
        best_confs = []
        if component in self.component_history:
            # Find all diagnostics within window that have this component in "✅" state
            for ts, state in self.component_history[component]:
                if window_start <= ts <= window_end and state == "✅":
                    # Find matching diagnostic result
                    for diag_ts, diag in self.diagnostic_history:
                        if abs(diag_ts - ts) < 0.5:  # Within 500ms window
                            comp_data = diag["truck_components"].get(component, {})
                            confs = comp_data.get("confidence", [])
                            best_confs.extend(confs)
        
        # Return top N confidences (N = expected count for component)
        expected_count = EXPECTED_COMPONENT_COUNTS.get(component, 1)
        return sorted(best_confs, reverse=True)[:expected_count]
                      
    def add_diagnostics(self, timestamp, diagnostics_result):
        """
        Add diagnostic result to history with automatic cleanup of old entries.
        
        Updates both full diagnostic history and per-component state history.
        Automatically sets start time if not previously initialized.
        
        Args:
            timestamp (float): Timestamp of diagnostic result in seconds (from time.time())
            diagnostics_result (dict): Diagnostic result dictionary with "truck_face" and "truck_components" keys
        """
        if self.processing_start_time is None:
            self.processing_start_time = timestamp
            
        # Store full diagnostic result
        self.diagnostic_history.append((timestamp, diagnostics_result))
        
        # Update component history
        for component, data in diagnostics_result.get("truck_components", {}).items():
            if component not in self.component_history:
                self.component_history[component] = []
            
            # Determine component state
            state = self._get_component_state(component, data)
            self.component_history[component].append((timestamp, state))

    def add_wiper_observation(self, track_id, side, timestamp, center_x, image_width):
        """
        Add wiper position observation for movement tracking.
        
        Args:
            track_id: YOLO track ID for the wiper
            side: "left" or "right" 
            timestamp: Frame timestamp (float)
            center_x: Center x-position of wiper bounding box
            image_width: Image width for relative threshold calculation
        """
        # Initialize wiper history if not exists
        if not hasattr(self, 'wiper_history'):
            self.wiper_history = {}
        
        key = f"{side}_wiper_{track_id}"
        
        # Reset history if track_id changes (new tracking instance)
        if key not in self.wiper_history:
            self.wiper_history[key] = {
                "side": side,
                "positions": [],
                "image_width": image_width
            }
        
        # Add new observation
        self.wiper_history[key]["positions"].append((timestamp, center_x))
        
        # Prune old entries beyond DIAGNOSTIC_INTERVAL_SECONDS window (e.g., 3 seconds)
        cutoff = timestamp - (WIPER_FRAMES_TO_COLLECT * WIPER_COLLECTION_INTERVAL_SECONDS)
        self.wiper_history[key]["positions"] = [
            (ts, cx) for ts, cx in self.wiper_history[key]["positions"]
            if ts >= cutoff
        ]

    def get_wiper_movement_status(self, track_id, side):
        """
        Check if wiper has moved significantly over tracking window.
        
        Returns:
            str: "moving", "stationary", or "insufficient_data"
        """
        if not hasattr(self, 'wiper_history'):
            return "insufficient_data", 0.0, 0.0, 0
        
        key = f"{side}_wiper_{track_id}"
        if key not in self.wiper_history:
            return "insufficient_data", 0.0, 0.0, 0
        
        history = self.wiper_history[key]
        positions = history["positions"]
        image_width = history["image_width"]
        frame_count = len(positions)

        # Check minimum frames requirement
        if len(positions) < WIPER_MIN_FRAMES_FOR_ANALYSIS:
            logger.debug(f"Wiper {side} (track_id={track_id}): insufficient frames ({frame_count} < {WIPER_MIN_FRAMES_FOR_ANALYSIS})")
            return "insufficient_data", 0.0, 0.0, frame_count
        
        # Calculate sweep range: max - min center-x over window
        center_x_values = [cx for _, cx in positions]
        sweep_range = max(center_x_values) - min(center_x_values)
        
        # Calculate relative threshold (10% of image width)
        threshold_pixels = image_width * WIPER_MOVEMENT_THRESHOLD_RELATIVE

        # Debug logging for movement verification
        logger.info(
            f"Wiper {side} (track_id={track_id}): "
            f"sweep_range={sweep_range:.1f}px, "
            f"threshold={threshold_pixels:.1f}px ({WIPER_MOVEMENT_THRESHOLD_RELATIVE*100:.0f}% of {image_width}px), "
            f"frames={len(positions)}, "
            f"status={'moving' if sweep_range >= threshold_pixels else 'stationary'}"
        )

        # Determine status
        if sweep_range >= threshold_pixels:
            return "moving", sweep_range, threshold_pixels, frame_count
        else:
            return "stationary", sweep_range, threshold_pixels, frame_count

    def add_light_observation(self, track_id, side, timestamp, crop_image):
        """
        Add light crop observation for brightness change analysis.
        
        Args:
            track_id: YOLO track ID for the light
            side: "left" or "right"
            timestamp: Frame timestamp (float)
            crop_image: NumPy array of light region (BGR format)
        """
        # Initialize light history if not exists
        if not hasattr(self, 'light_history'):
            self.light_history = {}
        
        key = f"{side}_light_front_{track_id}"
        
        # Reset history if track_id changes (new tracking instance)
        if key not in self.light_history:
            self.light_history[key] = {
                "side": side,
                "crops": [],  # List of (timestamp, crop_image)
                "grid_rows": LIGHT_GRID_ROWS,
                "grid_cols": LIGHT_GRID_COLS
            }
        
        # Add new observation
        self.light_history[key]["crops"].append((timestamp, crop_image))
        
        # Prune old entries beyond collection window (8 frames × 0.5s = 4s)
        cutoff = timestamp - (LIGHT_FRAMES_TO_COLLECT * LIGHT_COLLECTION_INTERVAL_SECONDS)
        self.light_history[key]["crops"] = [
            (ts, crop) for ts, crop in self.light_history[key]["crops"]
            if ts >= cutoff
        ]

    def _group_neighboring_changing_grids(self, changing_grid_mask, grid_rows, grid_cols):
        """
        Group neighboring "changing" grid cells into logical sections using connected components.
        
        Args:
            changing_grid_mask: 2D numpy array (grid_rows × grid_cols) with 1=changing, 0=static
            grid_rows: Number of rows in grid
            grid_cols: Number of columns in grid
        
        Returns:
            list: List of section dicts, each containing:
                - 'grid_positions': List of (row, col) tuples in this section
                - 'total_grids': Total grid cells in section
                - 'avg_brightness_std': Average std dev across all grids in section
        """  
        sections = []
        visited = np.zeros((grid_rows, grid_cols), dtype=bool)
        
        # 8-direction connectivity (includes diagonals for better grouping)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # BFS to find connected components
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Skip if already visited or not changing
                if visited[row, col] or changing_grid_mask[row, col] == 0:
                    continue
                
                # Found new section - start BFS to find all connected grids
                section_grids = []
                queue = deque([(row, col)])
                visited[row, col] = True
                
                while queue:
                    r, c = queue.popleft()
                    section_grids.append((r, c))
                    
                    # Check all 8 neighbors
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        # Check bounds
                        if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                            # Check if neighbor is changing and not visited
                            if not visited[nr, nc] and changing_grid_mask[nr, nc] == 1:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                
                sections.append({
                    'grid_positions': section_grids,
                    'total_grids': len(section_grids)
                })
        
        return sections

    def get_light_working_sections(self, track_id, side):
        """
        Analyze brightness changes across collected light crops to identify working sections.
        Groups neighboring changing grids into logical sections (1-3 expected for front lights).

        Returns:
            tuple: (status, section_count, total_grids, section_grid_counts, section_brightness_values, frame_count)
                - status: "lighting", "broken", or "insufficient_data"         # Indicates if light is working based on brightness changes
                - section_count: Number of grouped sections with changes       # Number of distinct sections identified based on brightness changes
                - total_grids: Total grids analyzed (grid_rows × grid_cols)    # Total number of grid cells analyzed across the light crop (grid_rows × grid_cols)
                - section_grid_counts: List of grid counts per section         # Number of grids in each identified section (e.g., [3, 2] for two sections with 3 and 2 grids respectively)
                - section_brightness_values: List of avg std_dev per section   # Average brightness std_dev for each identified section (e.g., [20.5, 18.3] for two sections with respective brightness changes)
                - frame_count: Number of frames analyzed
        """
        if not hasattr(self, 'light_history'):
            return "insufficient_data", 0, 0, [], [], 0
        
        key = f"{side}_light_front_{track_id}"
        if key not in self.light_history:
            return "insufficient_data", 0, 0, [], [], 0
        
        history = self.light_history[key]
        crops = history["crops"]
        frame_count = len(crops)
        
        # Check minimum frames requirement (follow wiper pattern)
        if frame_count < LIGHT_MIN_FRAMES_FOR_ANALYSIS:
            logger.debug(f"Light {side} (track_id={track_id}): insufficient frames ({frame_count} < {LIGHT_MIN_FRAMES_FOR_ANALYSIS})")
            return "insufficient_data", 0, 0, [], [], frame_count
        
        # Extract crop images (ignore timestamps for analysis)
        crop_images = [crop for _, crop in crops]
        
        # Resize all crops to consistent dimensions for grid analysis
        TARGET_H, TARGET_W = 128, 128  # Fixed size for reliable grid comparison
        resized_crops = []
        for crop in crop_images:
            if crop.shape[:2] != (TARGET_H, TARGET_W):
                # Resize with linear interpolation for smooth scaling
                resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                resized_crops.append(resized)
            else:
                resized_crops.append(crop)
        crop_images = resized_crops
        
        # Now all crops have guaranteed consistent dimensions
        h, w = crop_images[0].shape[:2]
        
        # Convert BGR to grayscale for brightness analysis
        gray_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for crop in crop_images]
        
        # Get grid dimensions
        grid_rows = LIGHT_GRID_ROWS
        grid_cols = LIGHT_GRID_COLS
        section_h = h // grid_rows
        section_w = w // grid_cols
        
        # Analyze each grid section and store std_dev values
        changing_grid_mask = np.zeros((grid_rows, grid_cols), dtype=int)
        grid_std_devs = {}  # Store std_dev for each grid cell
        total_grids = grid_rows * grid_cols
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Extract section coordinates
                y1 = row * section_h
                y2 = (row + 1) * section_h if row < grid_rows - 1 else h
                x1 = col * section_w
                x2 = (col + 1) * section_w if col < grid_cols - 1 else w
                
                # Compute mean brightness for this section across all frames
                section_brightness = []
                for gray in gray_crops:
                    section = gray[y1:y2, x1:x2]
                    mean_brightness = np.mean(section)
                    section_brightness.append(mean_brightness)
                
                # Compute standard deviation of brightness across frames
                brightness_std = np.std(section_brightness)
                grid_std_devs[(row, col)] = brightness_std
                
                # If std dev > threshold, section is "changing" (working)
                if brightness_std > LIGHT_BRIGHTNESS_CHANGE_THRESHOLD:
                    changing_grid_mask[row, col] = 1
        
        # Group neighboring changing grids into sections using connected components (BFS)
        sections = self._group_neighboring_changing_grids(changing_grid_mask, grid_rows, grid_cols)
        
        # Calculate avg brightness std for each section
        for section in sections:
            section['avg_brightness_std'] = np.mean([grid_std_devs[grid_pos] for grid_pos in section['grid_positions']])
        
        # Filter by minimum size threshold
        valid_sections = [s for s in sections if s['total_grids'] >= LIGHT_MIN_SECTION_GRIDS]
        
        # Sort by size (largest first)
        valid_sections.sort(key=lambda s: s['total_grids'], reverse=True)
        
        # Extract values for return
        section_grid_counts = [s['total_grids'] for s in valid_sections]
        section_brightness_values = [s['avg_brightness_std'] for s in valid_sections]
        section_count = len(valid_sections)

        # Determine status based on section count
        if section_count > 0:
            status = "lighting"
        else:
            status = "broken"

        # Log analysis results for debugging
        logger.info(
            f"Light {side} (track_id={track_id}): "
            f"analyzed {total_grids} grids across {frame_count} frames, "
            f"{section_count} section(s) working"
        )
        
        return status, section_count, total_grids, section_grid_counts, section_brightness_values, frame_count

    def get_consensus_diagnostics(self):
        """
        Compute consensus diagnostic messages from all collected samples (excluding ignore period).
        
        Applies consensus voting across collected diagnostics (up to COUNT_TO_DECIDE_CONSENSUS, after ignore period):
    - Plate numbers: Two-stage consensus using YOLO confidence tiebreaker
    - Front lights: Two-stage consensus (STATE → SECTION COUNT) using YOLO confidence tiebreaker
    - Other components: Single-stage consensus on state (✅/⚠️/❌) with confidence tiebreaker
        
        Returns:
            tuple: (consensus_diagnostics, truck_face, consensus_components, diagnostics_ok, diagnostics_ng)
                - consensus_diagnostics: List of consensus diagnostic messages
                - truck_face: Most recent truck face from usable window ("truck_front" or "truck_back")
                - consensus_components: Dictionary of consensus component states and data
                - diagnostics_ok: List of passing component messages (✅)
                - diagnostics_ng: List of failing component messages (❌/⚠️/❓)
        """
        if not self.diagnostic_history or self.processing_start_time is None:
            # Always return 5 values even in error case to prevent unpacking errors
            return None, None, None, [], []
        
        # Calculate usable time window (after ignore period)
        current_time = datetime.now().timestamp()
        ignore_cutoff = self.processing_start_time + self.ignore_period
        
        # Filter diagnostic history to usable window by ignore period only (no time window)
        usable_diagnostics = [
            (ts, diag) for ts, diag in self.diagnostic_history
            if ts >= ignore_cutoff
        ]
        
        if not usable_diagnostics:
            # No usable data, return latest available
            latest_diag = self.diagnostic_history[-1][1]
            diagnostics_ok = [m for m in latest_diag["diagnostics"] if m.startswith("✅")]
            diagnostics_ng = [m for m in latest_diag["diagnostics"] if m.startswith(("❌", "⚠️"))]
            return (
                latest_diag["diagnostics"],
                latest_diag["truck_face"],
                latest_diag["truck_components"],
                diagnostics_ok,
                diagnostics_ng
            )

        # Log how many samples were collected for consensus
        sample_count = len(usable_diagnostics)
        if sample_count < self.count_to_decide_consensus:
            logger.info(f"Consensus based on {sample_count} diagnostic sample(s) (video ended before {self.count_to_decide_consensus} collected)")
        else:
            logger.info(f"Consensus based on {sample_count} diagnostic sample(s)")

        # Use the most recent truck face from usable data
        latest_usable_diag = usable_diagnostics[-1][1]
        truck_face = latest_usable_diag["truck_face"]
        
        # Build consensus component states from usable window only
        consensus_components = {}
        consensus_diagnostics = []
        
        # Determine which components to check based on truck face
        expected_components = self._get_expected_components(truck_face)
        
        for component in expected_components:
            # PLATE NUMBER - 2-stage consensus (STATE → NUMBER) with YOLO confidence tiebreaker
            if component == "plate_number":
                # === LEVEL 1: Consensus on PLATE STATE (✅/⚠️/❌) ===
                state_occurrences = {}      # {state: count}
                state_yolo_confidences = {} # {state: [YOLO confidences]}

                for ts, diag in usable_diagnostics:
                    comp_data = diag["truck_components"].get("plate_number", {})
                    state = self._get_component_state("plate_number", comp_data)
                    yolo_confs = comp_data.get("confidence", [])
                    
                    # Track state occurrences
                    state_occurrences[state] = state_occurrences.get(state, 0) + 1
                    
                    # Track YOLO confidences for tiebreaking
                    if state not in state_yolo_confidences:
                        state_yolo_confidences[state] = []
                    state_yolo_confidences[state].extend(yolo_confs)
                
                # Apply two-stage consensus on STATES
                consensus_state = self._resolve_consensus_two_stage(
                    state_occurrences,
                    state_yolo_confidences  # YOLO confidence ONLY for tiebreaking
                )

                # --- Plate STATE consensus ---
                display_name = self._get_display_name("plate_number")
                logger.info(f"{display_name} STATE consensus analysis (window: {sample_count} frames, two-stage voting):")

                # Log candidates sorted by occurrence (descending), then by state for deterministic order
                sorted_states = sorted(
                    [(state, count) for state, count in state_occurrences.items() if count > 0],
                    key=lambda x: (-x[1], x[0])
                )
                for state, count in sorted_states:
                    confs = state_yolo_confidences.get(state, [])
                    avg_yolo = sum(confs) / len(confs) if confs else 0.0
                    logger.info(f"{display_name} STATE candidate '{state}': occurrences={count}, avg_yolo={avg_yolo:.2f}")
                
                winner_confs = state_yolo_confidences.get(consensus_state, [])
                winner_avg_yolo = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0
                logger.info(f"{display_name} STATE consensus winner: '{consensus_state}' (occurrences={state_occurrences.get(consensus_state, 0)}, avg_yolo={winner_avg_yolo:.2f})")
                logger.info("")  # Blank line separator
    
                # === EARLY EXIT: Skip number consensus if state ≠ ✅ ===
                if consensus_state != "✅":
                    # Build component with consensus state but NO number
                    consensus_components["plate_number"] = {
                        "count": 0,
                        "confidence": [],
                        "consensus_state": consensus_state
                    }
                    consensus_diagnostics.append(self._build_consensus_message("plate_number", consensus_state))
                    continue  # Skip to next component

                # === LEVEL 2: Consensus on PLATE NUMBER (ONLY from ✅ frames) ===
                plate_occurrences = {}          # {normalized_plate: count}
                plate_yolo_confidences = {}     # {normalized_plate: [list of YOLO detection confidences]}
                plate_raw_values = {}           # {normalized_plate: original_raw_string}
                
                # CRITICAL: ONLY process frames where plate state = ✅
                valid_frame_count = 0
                for ts, diag in usable_diagnostics:
                    comp_data = diag["truck_components"].get("plate_number", {})
                    state = self._get_component_state("plate_number", comp_data)
        
                    # FILTER: Skip non-✅ states (this is the missing constraint!)
                    if state != "✅":
                        continue
                    
                    valid_frame_count += 1
                    plate_num = comp_data.get("number")
                    yolo_confs = comp_data.get("confidence", [])

                    if plate_num:
                        normalized = self._normalize_plate_number(plate_num)
                        if normalized:
                            # Track occurrences (Stage 1)
                            plate_occurrences[normalized] = plate_occurrences.get(normalized, 0) + 1
                            
                            # Track YOLO confidences for tiebreaking (Stage 2) and output
                            if normalized not in plate_yolo_confidences:
                                plate_yolo_confidences[normalized] = []
                            if yolo_confs:
                                plate_yolo_confidences[normalized].extend(yolo_confs)

                            # Preserve one raw value for output
                            if normalized not in plate_raw_values:
                                plate_raw_values[normalized] = plate_num

                # --- LOGGING: Plate NUMBER consensus ---
                logger.info(f"Plate NUMBER consensus analysis (window: {valid_frame_count} frames, two-stage voting):")
                
                # Log candidates sorted by occurrence (descending), then by plate string
                sorted_plates = sorted(
                    [(plate, count) for plate, count in plate_occurrences.items() if count > 0],
                    key=lambda x: (-x[1], x[0])
                )
                for plate, count in sorted_plates:
                    confs = plate_yolo_confidences.get(plate, [])
                    avg_yolo = sum(confs) / len(confs) if confs else 0.0
                    logger.info(f"Plate NUMBER candidate '{plate}': occurrences={count}, avg_yolo={avg_yolo:.2f}")
                
                # Apply two-stage consensus on PLATE NUMBERS (occurrences → YOLO confidence)
                best_normalized_plate = self._resolve_consensus_two_stage(
                    plate_occurrences,
                    plate_yolo_confidences  # YOLO confidence ONLY (NO OCR)
                )

                if best_normalized_plate:
                    # Log winner selection details (occurrences + YOLO confidence)
                    winner_occurrences = plate_occurrences[best_normalized_plate]
                    winner_yolo = plate_yolo_confidences.get(best_normalized_plate, [0.95])
                    avg_yolo_winner = sum(winner_yolo) / len(winner_yolo) if winner_yolo else 0.95
                    
                    logger.info(f"Plate NUMBER consensus winner: '{best_normalized_plate}' (occurrences={winner_occurrences}, avg_yolo={avg_yolo_winner:.2f})")
                    logger.info("")  # Blank line separator
        
                    # Build consensus component with number
                    consensus_components["plate_number"] = {
                        "count": 1,
                        "confidence": [round(avg_yolo_winner, 2)],
                        "consensus_state": "✅",
                        "number": plate_raw_values[best_normalized_plate]
                    }
                    consensus_diagnostics.append(
                        f"✅ plate number: visible, with number: {plate_raw_values[best_normalized_plate]}"
                    )
                else:
                    # Edge case: state=✅ but no valid numbers extracted
                    logger.info("Plate NUMBER consensus winner: None (no valid plate numbers in ✅ frames)")
                    logger.info("")  # Blank line separator
                    
                    consensus_components["plate_number"] = {
                        "count": 0,
                        "confidence": [],
                        "consensus_state": "⚠️"
                    }
                    consensus_diagnostics.append("⚠️ plate number: visible but could not be read")

            # FRONT LIGHTS - 2-stage consensus (STATE → SECTION COUNT) with YOLO confidence tiebreaker
            elif component in ("left_light_front", "right_light_front"):
                # === LEVEL 1: Consensus on LIGHT STATE (✅/❌) ===
                state_occurrences = {}      # {state: count}
                state_yolo_confidences = {} # {state: [YOLO confidences]}
                for ts, diag in usable_diagnostics:
                    comp_data = diag["truck_components"].get(component, {})
                    state = self._get_component_state(component, comp_data)
                    yolo_confs = comp_data.get("confidence", [])
                    # Track state occurrences
                    state_occurrences[state] = state_occurrences.get(state, 0) + 1
                    # Track YOLO confidences for tiebreaking
                    if state not in state_yolo_confidences:
                        state_yolo_confidences[state] = []
                    state_yolo_confidences[state].extend(yolo_confs)
                
                # Apply two-stage consensus on STATES
                consensus_state = self._resolve_consensus_two_stage(
                    state_occurrences,
                    state_yolo_confidences  # YOLO confidence ONLY for tiebreaking
                )
                
                # --- Light STATE consensus ---
                display_name = self._get_display_name(component)
                logger.info(f"{display_name} STATE consensus analysis (window: {sample_count} frames, two-stage voting):")
                # Log candidates sorted by occurrence (descending), then by state
                sorted_states = sorted(
                    [(state, count) for state, count in state_occurrences.items() if count > 0],
                    key=lambda x: (-x[1], x[0])
                )
                for state, count in sorted_states:
                    confs = state_yolo_confidences.get(state, [])
                    avg_yolo = sum(confs) / len(confs) if confs else 0.0
                    logger.info(f"{display_name} STATE candidate '{state}': occurrences={count}, avg_yolo={avg_yolo:.2f}")
                winner_confs = state_yolo_confidences.get(consensus_state, [])
                winner_avg_yolo = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0
                logger.info(f"{display_name} STATE consensus winner: '{consensus_state}' (occurrences={state_occurrences.get(consensus_state, 0)}, avg_yolo={winner_avg_yolo:.2f})")
                logger.info("")  # Blank line separator

                # === EARLY EXIT: Skip section consensus if state ≠ ✅ ===
                if consensus_state != "✅":
                    # Build component with consensus state but NO section count
                    consensus_components[component] = {
                        "count": 0,
                        "confidence": [],
                        "consensus_state": consensus_state
                    }
                    consensus_diagnostics.append(self._build_consensus_message(component, consensus_state))
                    continue  # Skip to next component
                
                # === LEVEL 2: Consensus on SECTION COUNT (ONLY from ✅ frames) ===
                section_occurrences = {}          # {section_count: count}
                section_yolo_confidences = {}     # {section_count: [list of YOLO detection confidences]}
                # CRITICAL: ONLY process frames where light state = ✅
                valid_frame_count = 0
                for ts, diag in usable_diagnostics:
                    comp_data = diag["truck_components"].get(component, {})
                    state = self._get_component_state(component, comp_data)
                    # FILTER: Skip non-✅ states
                    if state != "✅":
                        continue
                    valid_frame_count += 1
                    # Get section count from component data (stored by parts_diagnostics.py)
                    section_count = comp_data.get("light_sections")
                    yolo_confs = comp_data.get("confidence", [])
                    if section_count is not None and section_count > 0:
                        # Track occurrences (Stage 1)
                        section_occurrences[section_count] = section_occurrences.get(section_count, 0) + 1
                        # Track YOLO confidences for tiebreaking (Stage 2)
                        if section_count not in section_yolo_confidences:
                            section_yolo_confidences[section_count] = []
                        if yolo_confs:
                            section_yolo_confidences[section_count].extend(yolo_confs)

                # --- LOGGING: Light SECTION consensus ---
                logger.info(f"{display_name} SECTION consensus analysis (window: {valid_frame_count} frames, two-stage voting):")
                # Log candidates sorted by occurrence (descending), then by section count
                sorted_sections = sorted(
                    [(sec, count) for sec, count in section_occurrences.items() if count > 0],
                    key=lambda x: (-x[1], x[0])
                )
                for sec, count in sorted_sections:
                    confs = section_yolo_confidences.get(sec, [])
                    avg_yolo = sum(confs) / len(confs) if confs else 0.0
                    logger.info(f"{display_name} SECTION candidate '{sec} sections': occurrences={count}, avg_yolo={avg_yolo:.2f}")
                
                # Apply two-stage consensus on SECTION COUNTS (occurrences → YOLO confidence)
                best_section_count = self._resolve_consensus_two_stage(
                    section_occurrences,
                    section_yolo_confidences  # YOLO confidence ONLY
                )
                
                if best_section_count is not None and best_section_count > 0:
                    # Log winner selection details (occurrences + YOLO confidence)
                    winner_occurrences = section_occurrences[best_section_count]
                    winner_yolo = section_yolo_confidences.get(best_section_count, [0.90])
                    avg_yolo_winner = sum(winner_yolo) / len(winner_yolo) if winner_yolo else 0.90
                    logger.info(f"{display_name} SECTION consensus winner: '{best_section_count} sections': occurrences={winner_occurrences}, avg_yolo={avg_yolo_winner:.2f}")
                    logger.info("")  # Blank line separator
                    # Build consensus component with section count
                    consensus_components[component] = {
                        "count": 1,
                        "confidence": [round(avg_yolo_winner, 2)],
                        "consensus_state": "✅",
                        "light_sections": best_section_count
                    }
                    # Concise consensus message (no grid/brightness details)
                    consensus_diagnostics.append(
                        f"✅ {component.replace('_', ' ')}: ok, with {best_section_count} sections working"
                    )
                else:
                    # Edge case: state=✅ but no valid section counts
                    logger.info(f"{display_name} SECTION consensus winner: None (no valid section counts in {valid_frame_count} ✅ frames)")
                    logger.info(f"  Debug: section_occurrences={section_occurrences}")
                    logger.info("")  # Blank line separator
                    consensus_components[component] = {
                        "count": 1,
                        "confidence": [0.90],
                        "consensus_state": "✅",
                        "light_sections": 1  # Default to 1 section
                    }
                    consensus_diagnostics.append(f"✅ {component.replace('_', ' ')}: ok, with 1 sections working")

            # REGULAR COMPONENTS: Single-stage consensus on states (✅/⚠️/❌)
            else:
                # --- REGULAR COMPONENTS: Two-stage consensus on states (✅/⚠️/❌) ---
                state_occurrences = {}      # {state: count}
                state_yolo_confidences = {}   # {state: [list of detection confidences]}
                
                for ts, diag in usable_diagnostics:
                    comp_data = diag["truck_components"].get(component, {})
                    state = self._get_component_state(component, comp_data)
                    yolo_confs = comp_data.get("confidence", [])
                    
                    # Track state occurrences
                    state_occurrences[state] = state_occurrences.get(state, 0) + 1
                    
                    # Track confidences for tiebreaker
                    if state not in state_yolo_confidences:
                        state_yolo_confidences[state] = []
                    state_yolo_confidences[state].extend(yolo_confs)

                # Apply two-stage consensus to select best state
                consensus_state = self._resolve_consensus_two_stage(
                    state_occurrences, 
                    state_yolo_confidences
                )
                
                if consensus_state is None:
                    consensus_state = "❓"

                # --- LOGGING: Component STATE consensus ---
                display_name = self._get_display_name(component)
                total_usable_frames = len(usable_diagnostics)
                
                logger.info(f"{display_name} STATE consensus analysis (window: {total_usable_frames} frames, two-stage voting):")
                
                # Log candidates sorted by occurrence (descending), then by state
                sorted_states = sorted(
                    [(state, count) for state, count in state_occurrences.items() if count > 0],
                    key=lambda x: (-x[1], x[0])
                )
                for state, count in sorted_states:
                    confs = state_yolo_confidences.get(state, [])
                    avg_yolo = sum(confs) / len(confs) if confs else 0.0
                    logger.info(f"{display_name} STATE candidate '{state}': occurrences={count}, avg_yolo={avg_yolo:.2f}")
                
                winner_confs = state_yolo_confidences.get(consensus_state, [])
                winner_avg_yolo = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0
                logger.info(f"{display_name} STATE consensus winner: '{consensus_state}' (occurrences={state_occurrences.get(consensus_state, 0)}, avg_yolo={winner_avg_yolo:.2f})")
                logger.info("")  # Blank line separator

                # Build component data based on consensus state
                if consensus_state == "✅":
                    # For ✅ state: use highest-confidence detections from frames that produced ✅ state
                    best_confs = []
                    for ts, diag in usable_diagnostics:
                        comp_data = diag["truck_components"].get(component, {})
                        if self._get_component_state(component, comp_data) == "✅":
                            confs = comp_data.get("confidence", [])
                            best_confs.extend(confs)
                    
                    # Keep top N confidences (N = expected count)
                    expected_count = EXPECTED_COMPONENT_COUNTS.get(component, 1)
                    best_confs = sorted(best_confs, reverse=True)[:expected_count]
                    
                    consensus_components[component] = {
                        "count": len(best_confs),
                        "confidence": [round(c, 2) for c in best_confs],
                        "consensus_state": "✅"
                    }
                elif consensus_state == "⚠️":
                    consensus_components[component] = {
                        "count": 1,
                        "confidence": [0.75],  # Representative mid-confidence
                        "consensus_state": "⚠️"
                    }
                else:  # "❌" or "❓"
                    consensus_components[component] = {
                        "count": 0,
                        "confidence": [],
                        "consensus_state": consensus_state
                    }

                # Build diagnostic message
                message = self._build_consensus_message(component, consensus_state)
                consensus_diagnostics.append(message)
        
        # Split diagnostics into OK and NG lists
        diagnostics_ok = [m for m in consensus_diagnostics if m.startswith("✅")]
        diagnostics_ng = [m for m in consensus_diagnostics if m.startswith(("❌", "⚠️", "❓"))]
        
        return consensus_diagnostics, truck_face, consensus_components, diagnostics_ok, diagnostics_ng