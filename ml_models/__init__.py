# ML Models Package for Smart Presence System
# This package contains face detection and emotion recognition models

__version__ = "1.0.0"
__author__ = "Smart Presence Team"

from .face_detection import FaceDetector
from .emotion_detection import EmotionClassifier
from .inference import SmartPresenceInference, get_inference_instance, recognize_student, process_attendance_image, check_models_available

__all__ = [
    "FaceDetector", 
    "EmotionClassifier", 
    "SmartPresenceInference",
    "get_inference_instance",
    "recognize_student",
    "process_attendance_image",
    "check_models_available"
] 