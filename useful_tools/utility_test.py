""" utility_test.py - Pure utility functions """

import logging
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

# --- GPU/CPU Detection ---
def get_device():
    """
    Automatically select GPU if available, otherwise fall back to CPU.
    
    Returns:
        str: Device identifier ('cuda' if GPU available, 'cpu' otherwise)
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU detected)")
    return device

# --- Qari-OCR Model Caching (mirrors YOLO caching pattern) ---
_qwen_model = None
_qwen_processor = None
_qwen_device = None

def get_cached_vlm_model():
    """
    Get cached Qari-OCR model instance to avoid reloading between frames.
    
    Returns:
        tuple: (model, processor, device)
            - model: Cached Qwen2VL model instance
            - processor: Associated processor for input preparation
            - device: Device where model is loaded ('cuda' or 'cpu')
    """
    global _qwen_model, _qwen_processor, _qwen_device
    if _qwen_model is None:
        model_name = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"
        _qwen_device = get_device()
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if _qwen_device == "cuda" else torch.float32,
            device_map="auto"
        )
        _qwen_processor = AutoProcessor.from_pretrained(
            model_name
            )
    return _qwen_model, _qwen_processor, _qwen_device