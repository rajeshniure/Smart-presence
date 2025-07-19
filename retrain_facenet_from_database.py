import os
import sys
import django
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import logging

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartPresence.settings')
django.setup()

from attendance.models import Student
from django.core.files.storage import default_storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
EMBEDDINGS_PATH = 'facenet_embeddings.npy'
LABELS_PATH = 'facenet_labels.npy'
CLASSIFIER_PATH = 'facenet_svm.joblib'
ENCODER_PATH = 'facenet_label_encoder.joblib'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def retrain_facenet_from_database():
    """Retrain FaceNet model using student photos from Django database"""
    
    logger.info("Starting FaceNet retraining from database...")
    
    # Load FaceNet model
    logger.info("Loading FaceNet models...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    
    # Get all students from database
    students = Student.objects.all()
    logger.info(f"Found {students.count()} students in database")
    
    if students.count() == 0:
        logger.error("No students found in database. Please add students first.")
        return False
    
    # Prepare data
    embeddings = []
    labels = []
    processed_count = 0
    failed_count = 0
    
    for student in tqdm(students, desc='Processing students'):
        try:
            # Check if student has an image
            if not student.image:
                logger.warning(f"Student {student.name} has no image, skipping...")
                failed_count += 1
                continue
            
            # Get the image path
            image_path = student.image.path
            
            # Check if file exists
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found for {student.name}: {image_path}")
                failed_count += 1
                continue
            
            # Load and process image
            img = Image.open(image_path).convert('RGB')
            
            # Detect and align face
            face = mtcnn(img)
            if face is not None:
                face = face.unsqueeze(0).to(device)
                emb = resnet(face).detach().cpu().numpy()[0]
                embeddings.append(emb)
                labels.append(student.name)
                processed_count += 1
                logger.info(f"Successfully processed {student.name}")
            else:
                logger.warning(f"No face detected in image for {student.name}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {student.name}: {str(e)}")
            failed_count += 1
            continue
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings generated. Cannot train model.")
        return False
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    logger.info(f"Generated {len(embeddings)} embeddings from {processed_count} students")
    logger.info(f"Failed to process {failed_count} students")
    
    # Save embeddings and labels
    logger.info("Saving embeddings and labels...")
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)
    
    # Encode labels
    logger.info("Encoding labels...")
    encoder = LabelEncoder()
    labels_num = encoder.fit_transform(labels)
    joblib.dump(encoder, ENCODER_PATH)
    
    # Train SVM classifier
    logger.info("Training SVM classifier...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, labels_num)
    joblib.dump(clf, CLASSIFIER_PATH)
    
    logger.info("FaceNet retraining completed successfully!")
    logger.info(f"Model files saved:")
    logger.info(f"  - Embeddings: {EMBEDDINGS_PATH}")
    logger.info(f"  - Labels: {LABELS_PATH}")
    logger.info(f"  - Classifier: {CLASSIFIER_PATH}")
    logger.info(f"  - Encoder: {ENCODER_PATH}")
    
    # Print summary
    unique_labels = np.unique(labels)
    logger.info(f"Model trained on {len(unique_labels)} unique students:")
    for label in unique_labels:
        count = np.sum(labels == label)
        logger.info(f"  - {label}: {count} embeddings")
    
    return True

def verify_model_files():
    """Verify that all model files exist and are valid"""
    required_files = [
        EMBEDDINGS_PATH,
        LABELS_PATH,
        CLASSIFIER_PATH,
        ENCODER_PATH
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            return False
    
    logger.info("All model files verified successfully!")
    return True

if __name__ == '__main__':
    try:
        success = retrain_facenet_from_database()
        if success:
            verify_model_files()
            print("\n✅ FaceNet retraining completed successfully!")
            print("🔄 Please restart the Django server to load the new models.")
        else:
            print("\n❌ FaceNet retraining failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1) 