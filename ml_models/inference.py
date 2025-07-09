import os
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import base64
from PIL import Image
import io

from .face_detection import FaceDetector
from .emotion_detection import EmotionClassifier

logger = logging.getLogger(__name__)

class SmartPresenceInference:
    """
    Unified inference class for Smart Presence System
    Handles both face detection and emotion recognition
    """
    
    def __init__(self, model_config_path="models/model_config.json"):
        self.face_detector = None
        self.emotion_classifier = None
        self.config = None
        self.models_loaded = False
        
        self.load_models(model_config_path)
    
    def load_models(self, config_path):
        """
        Load both face detection and emotion recognition models
        """
        try:
            # Load configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"Config file not found: {config_path}")
                self.config = self._default_config()
            
            # Load face detection model
            face_model_path = self.config['face_detection']['model_path']
            if os.path.exists(face_model_path):
                self.face_detector = FaceDetector(model_path=face_model_path)
                logger.info("Face detection model loaded successfully")
            else:
                logger.warning(f"Face detection model not found: {face_model_path}")
            
            # Load emotion recognition model
            emotion_model_path = self.config['emotion_recognition']['model_path']
            if os.path.exists(emotion_model_path):
                self.emotion_classifier = EmotionClassifier(model_path=emotion_model_path)
                logger.info("Emotion recognition model loaded successfully")
            else:
                logger.warning(f"Emotion recognition model not found: {emotion_model_path}")
            
            self.models_loaded = (self.face_detector is not None and 
                                self.emotion_classifier is not None)
            
            if self.models_loaded:
                logger.info("All models loaded successfully")
            else:
                logger.warning("Some models failed to load")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
    
    def _default_config(self):
        """
        Default configuration if config file is not found
        """
        return {
            "face_detection": {
                "model_path": "models/face_detection_model.h5",
                "input_size": [224, 224],
                "confidence_threshold": 0.5
            },
            "emotion_recognition": {
                "model_path": "models/emotion_detection_model.h5",
                "input_size": [48, 48],
                "emotion_labels": ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
            }
        }
    
    def process_image(self, image_input: Union[str, np.ndarray, bytes]) -> Dict:
        """
        Process an image for both face detection and emotion recognition
        
        Args:
            image_input: Can be file path, numpy array, or base64 encoded bytes
            
        Returns:
            Dict containing face detection and emotion recognition results
        """
        if not self.models_loaded:
            return {
                'success': False,
                'error': 'Models not loaded properly',
                'face_detected': False,
                'emotion': None
            }
        
        try:
            # Convert input to opencv image format
            image = self._process_image_input(image_input)
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not process input image',
                    'face_detected': False,
                    'emotion': None
                }
            
            # Face detection
            face_result = self.face_detector.predict(image)
            
            # Check if face was detected with sufficient confidence
            face_detected = face_result['confidence'] >= self.config['face_detection']['confidence_threshold']
            
            result = {
                'success': True,
                'face_detected': face_detected,
                'face_confidence': face_result['confidence'],
                'face_bbox': face_result['bbox'] if face_detected else None,
                'emotion': None,
                'emotion_confidence': 0.0,
                'emotion_scores': {}
            }
            
            # If face detected, perform emotion recognition
            if face_detected:
                # Extract face region for emotion recognition
                x1, y1, x2, y2 = face_result['bbox']
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                if x2 > x1 and y2 > y1:
                    face_roi = image[y1:y2, x1:x2]
                    
                    # Emotion recognition
                    emotion_result = self.emotion_classifier.predict(face_roi)
                    
                    result.update({
                        'emotion': emotion_result['emotion'],
                        'emotion_confidence': emotion_result['confidence'],
                        'emotion_scores': emotion_result['all_scores']
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'success': False,
                'error': str(e),
                'face_detected': False,
                'emotion': None
            }
    
    def _process_image_input(self, image_input: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
        """
        Convert various image input formats to OpenCV format
        """
        try:
            if isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Base64 encoded image
                    header, data = image_input.split(',', 1)
                    image_data = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_data))
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                else:
                    # File path
                    image = cv2.imread(image_input)
            elif isinstance(image_input, bytes):
                # Raw image bytes
                image = Image.open(io.BytesIO(image_input))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                # Already in correct format
                image = image_input
            else:
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image input: {e}")
            return None
    
    def process_webcam_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a webcam frame for real-time recognition
        Optimized for speed
        """
        return self.process_image(frame)
    
    def batch_process_images(self, image_list: List[Union[str, np.ndarray]]) -> List[Dict]:
        """
        Process multiple images in batch
        """
        results = []
        for image in image_list:
            result = self.process_image(image)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        """
        return {
            'models_loaded': self.models_loaded,
            'face_detector_loaded': self.face_detector is not None,
            'emotion_classifier_loaded': self.emotion_classifier is not None,
            'config': self.config
        }
    
    def is_ready(self) -> bool:
        """
        Check if the inference system is ready for predictions
        """
        return self.models_loaded

# Global inference instance
_inference_instance = None

def get_inference_instance() -> SmartPresenceInference:
    """
    Get singleton inference instance
    """
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = SmartPresenceInference()
    return _inference_instance

def recognize_student(image_input: Union[str, np.ndarray, bytes]) -> Dict:
    """
    High-level function to recognize a student from an image
    
    Returns:
        Dict with recognition results including face detection and emotion
    """
    inference = get_inference_instance()
    return inference.process_image(image_input)

def process_attendance_image(image_data: str) -> Dict:
    """
    Process base64 encoded image from webcam for attendance
    
    Args:
        image_data: Base64 encoded image string from webcam
        
    Returns:
        Dict with face detection and emotion recognition results
    """
    inference = get_inference_instance()
    result = inference.process_image(image_data)
    
    # Add additional attendance-specific processing
    if result['success'] and result['face_detected']:
        # Calculate attendance status based on current time
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        if hour < 9 or (hour == 9 and minute <= 10):
            status = 'present'
        elif hour < 10:
            status = 'late'
        else:
            status = 'absent'
        
        result['attendance_status'] = status
        result['timestamp'] = now.isoformat()
    
    return result

def check_models_available() -> Dict:
    """
    Check if trained models are available
    """
    config_path = "models/model_config.json"
    face_model_path = "models/face_detection_model.h5"
    emotion_model_path = "models/emotion_detection_model.h5"
    
    return {
        'config_exists': os.path.exists(config_path),
        'face_model_exists': os.path.exists(face_model_path),
        'emotion_model_exists': os.path.exists(emotion_model_path),
        'all_models_available': all([
            os.path.exists(config_path),
            os.path.exists(face_model_path),
            os.path.exists(emotion_model_path)
        ])
    } 