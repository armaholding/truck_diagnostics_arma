# diagnostic_history.py
"""Temporal consensus engine for truck component diagnostics with yolo confidence tie breaker."""

import logging
from datetime import datetime
import re
from config import (
    DIAGNOSTIC_THRESHOLD,
    EXPECTED_COMPONENT_COUNTS,
    FRONT_EXPECTED_COMPONENTS,
    BACK_EXPECTED_COMPONENTS
)

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
    
    def __init__(self, consensus_window_seconds=18, ignore_period_seconds=1):
        """
        Initialize diagnostic history tracker with temporal window configuration.
        
        Args:
            consensus_window_seconds (int): Duration in seconds for temporal consensus window (default: 18)
            ignore_period_seconds (int): Duration in seconds to ignore at start of processing for stabilization (default: 1)
        """
        self.consensus_window = consensus_window_seconds
        self.ignore_period = ignore_period_seconds
        self.diagnostic_history = []  # List of (timestamp, diagnostics_dict)
        self.component_history = {}   # Track each component's states over time
        self.processing_start_time = None

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
            data (dict): Component detection data with "count" and "confidence" keys
        
        Returns:
            str: State emoji ("✅" = pass, "⚠️" = detected but unreadable, "❌" = fail)
        """
        count = data.get("count", 0)
        confidences = data.get("confidence", [])
        
        if component == "plate_number":
            # Plate number state depends on OCR success
            if "number" in data:
                return "✅"
            elif count >= EXPECTED_COMPONENT_COUNTS.get(component, 1) and all(c >= DIAGNOSTIC_THRESHOLD for c in confidences):
                return "⚠️"
            else:
                return "❌"
        
        # For other components
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
    
    def _cleanup_old_entries(self, current_time):
        """
        Remove diagnostic history entries older than consensus window boundary.
        
        Maintains sliding window by pruning entries before (current_time - consensus_window).
        
        Args:
            current_time (float): Current timestamp in seconds (from time.time())
        """
        cutoff_time = current_time - self.consensus_window
        self.diagnostic_history = [
            (ts, diag) for ts, diag in self.diagnostic_history 
            if ts >= cutoff_time
        ]
        
        for component in self.component_history:
            self.component_history[component] = [
                (ts, state) for ts, state in self.component_history[component]
                if ts >= cutoff_time
            ]

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
        
        # Clean up old entries (keep full consensus window)
        self._cleanup_old_entries(timestamp)
    
    def get_consensus_diagnostics(self):
        """
        Compute consensus diagnostic messages from history (excluding ignore period).
        
        Applies temporal voting across 18-second window (after 1-second ignore period):
        - Plate numbers: Two-stage consensus using weighted score (0.60*YOLO + 0.40*OCR)
        - Components: Two-stage consensus on state (✅/⚠️/❌) with confidence tiebreaker
        
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
        window_end = current_time
        window_start = max(ignore_cutoff, current_time - self.consensus_window)
        
        # Filter diagnostic history to usable window
        usable_diagnostics = [
            (ts, diag) for ts, diag in self.diagnostic_history
            if ts >= window_start and ts <= window_end
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
        
        # Use the most recent truck face from usable data
        latest_usable_diag = usable_diagnostics[-1][1]
        truck_face = latest_usable_diag["truck_face"]
        
        # Build consensus component states from usable window only
        consensus_components = {}
        consensus_diagnostics = []
        
        # Determine which components to check based on truck face
        expected_components = self._get_expected_components(truck_face)
        
        for component in expected_components:
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
                total_usable_frames = len(usable_diagnostics)
                
                logger.info(f"{display_name} STATE consensus analysis (window: {total_usable_frames} frames, two-stage voting):")

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

                # --- ENHANCED LOGGING: Plate NUMBER consensus ---
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

                # --- ENHANCED LOGGING: Component STATE consensus ---
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