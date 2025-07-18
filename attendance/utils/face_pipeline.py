import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models with error handling
yolo_model = None
resnet = None
mtcnn = None
clf = None
encoder = None

def load_models():
    """Load all models with error handling"""
    global yolo_model, resnet, mtcnn, clf, encoder
    
    try:
        # Load YOLO model
        yolo_path = 'yolo_runs/face_yolo/weights/best.pt'
        if not os.path.exists(yolo_path):
            logger.error(f"YOLO model not found at: {yolo_path}")
            # Try alternative paths
            alternative_paths = [
                'yolo_runs/face_yolo2/weights/best.pt',
                'yolo_runs/face_yolo/best.pt',
                'best.pt'
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    yolo_path = path
                    logger.info(f"Using YOLO model from: {path}")
                    break
            else:
                logger.error("No YOLO model found. Please train the model first.")
                return False
        
        yolo_model = YOLO(yolo_path)
        logger.info("YOLO model loaded successfully")
        
        # Load FaceNet models
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        mtcnn = MTCNN(image_size=160, margin=0)
        logger.info("FaceNet models loaded successfully")
        
        # Load SVM classifier and encoder
        clf_path = 'facenet_svm.joblib'
        encoder_path = 'facenet_label_encoder.joblib'
        
        if not os.path.exists(clf_path) or not os.path.exists(encoder_path):
            logger.error("SVM classifier or label encoder not found. Please train FaceNet first.")
            return False
        
        clf = joblib.load(clf_path)
        encoder = joblib.load(encoder_path)
        logger.info("SVM classifier and label encoder loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Load models on import
models_loaded = load_models()

def detect_and_recognize(image_path):
    """
    Given an image path, detect faces with YOLO, recognize with FaceNet+SVM.
    Returns: list of dicts: [{'box': [x1, y1, x2, y2], 'name': 'person'}]
    """
    if not models_loaded:
        logger.error("Models not loaded. Cannot perform detection/recognition.")
        return []
    
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return []
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Processing image: {image_path}, size: {image.size}")
        
        # YOLO face detection
        try:
            results = yolo_model(image)
            logger.info(f"YOLO detection completed. Found {len(results[0].boxes) if results[0].boxes is not None else 0} faces")
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}")
            return []
        
        faces = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    logger.info(f"Processing face {i+1}: box={[x1, y1, x2, y2]}")
                    
                    # Ensure valid coordinates
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.width, x2)
                    y2 = min(image.height, y2)
                    
                    # Check if face region is valid
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"Invalid face region for face {i+1}: {[x1, y1, x2, y2]}")
                        continue
                    
                    # Crop face
                    face_img = image.crop((x1, y1, x2, y2))
                    logger.info(f"Face {i+1} cropped to size: {face_img.size}")
                    
                    # FaceNet recognition with multiple attempts
                    name = 'Unknown'
                    face_tensor = None
                    
                    # Try different approaches for MTCNN
                    try:
                        # Method 1: Direct MTCNN processing
                        face_tensor = mtcnn(face_img)
                        if face_tensor is not None:
                            logger.info(f"Face {i+1} processed successfully by MTCNN")
                        else:
                            logger.warning(f"Face {i+1} returned None from MTCNN")
                    except Exception as e:
                        logger.warning(f"MTCNN failed for face {i+1}: {str(e)}")
                    
                    # If MTCNN failed, try resizing the image
                    if face_tensor is None:
                        try:
                            # Resize to a reasonable size for MTCNN
                            face_img_resized = face_img.resize((160, 160), Image.LANCZOS)
                            face_tensor = mtcnn(face_img_resized)
                            if face_tensor is not None:
                                logger.info(f"Face {i+1} processed successfully after resizing")
                        except Exception as e:
                            logger.warning(f"MTCNN failed even after resizing for face {i+1}: {str(e)}")
                    
                    # If still failed, try with different MTCNN settings
                    if face_tensor is None:
                        try:
                            # Create a new MTCNN instance with different settings
                            mtcnn_alt = MTCNN(image_size=160, margin=10, keep_all=False)
                            face_tensor = mtcnn_alt(face_img)
                            if face_tensor is not None:
                                logger.info(f"Face {i+1} processed successfully with alternative MTCNN")
                        except Exception as e:
                            logger.warning(f"Alternative MTCNN also failed for face {i+1}: {str(e)}")
                    
                    # Process the face tensor if available
                    if face_tensor is not None:
                        try:
                            emb = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
                            pred = clf.predict(emb)[0]
                            name = encoder.inverse_transform([pred])[0]
                            logger.info(f"Face {i+1} recognized as: {name}")
                        except Exception as e:
                            logger.error(f"Error in FaceNet recognition for face {i+1}: {str(e)}")
                            name = 'Unknown'
                    else:
                        logger.warning(f"Face {i+1} could not be processed by any MTCNN method")
                        name = 'Unknown'
                    
                    faces.append({'box': [x1, y1, x2, y2], 'name': name})
                    
                except Exception as e:
                    logger.error(f"Error processing face {i+1}: {str(e)}")
                    continue
        else:
            logger.info("No faces detected by YOLO")
        
        logger.info(f"Detection/recognition completed. Found {len(faces)} faces")
        return faces
        
    except Exception as e:
        logger.error(f"Error in detect_and_recognize: {str(e)}")
        return []

# Test function for debugging
def test_pipeline():
    """Test the pipeline with a sample image"""
    logger.info("Testing face detection and recognition pipeline...")
    
    # Check if models are loaded
    if not models_loaded:
        logger.error("Models not loaded. Cannot test pipeline.")
        return False
    
    # Look for a test image
    test_image_paths = [
        'datasets/face_recognition',
        'media/student_images',
        'static/img'
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(path, file)
                    break
            if test_image:
                break
    
    if test_image:
        logger.info(f"Testing with image: {test_image}")
        results = detect_and_recognize(test_image)
        logger.info(f"Test results: {results}")
        return len(results) > 0
    else:
        logger.warning("No test image found")
        return False 