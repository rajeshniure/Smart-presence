"""Utility package for computer vision and model management.

Currently provides a lightweight face detection pipeline using OpenCV.
Recognition is stubbed to return "Unknown" until a proper model is trained.
"""

# Re-export commonly used functions for convenience
from .face_pipeline import detect_and_recognize, reload_models  # noqa: F401


