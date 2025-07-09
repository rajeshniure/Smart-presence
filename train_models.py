#!/usr/bin/env python3
"""
Smart Presence System - Model Training Script

This script trains both face detection and emotion recognition models
using the provided datasets.

Usage:
    python train_models.py --face --emotion --epochs 100

Arguments:
    --face: Train face detection model
    --emotion: Train emotion recognition model
    --epochs: Number of training epochs (default: 100)
    --batch-size: Batch size for training (default: 32)
    --learning-rate: Learning rate (default: 0.001)
"""

import argparse
import os
import sys
import logging
import time
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.face_detection import FaceDetector, train_face_detection_model
from ml_models.emotion_detection import EmotionClassifier, train_emotion_detection_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_datasets():
    """
    Check if datasets are available and properly structured
    """
    logger.info("Checking dataset availability...")
    
    # Check face detection dataset
    face_train_images = "datasets/face detection/images/train"
    face_train_labels = "datasets/face detection/labels/train"
    face_val_images = "datasets/face detection/images/val"
    face_val_labels = "datasets/face detection/labels/val"
    
    face_dataset_available = all([
        os.path.exists(face_train_images),
        os.path.exists(face_train_labels),
        os.path.exists(face_val_images),
        os.path.exists(face_val_labels)
    ])
    
    if face_dataset_available:
        train_images_count = len([f for f in os.listdir(face_train_images) if f.endswith('.jpg')])
        train_labels_count = len([f for f in os.listdir(face_train_labels) if f.endswith('.txt')])
        logger.info(f"Face detection dataset: {train_images_count} training images, {train_labels_count} labels")
    else:
        logger.warning("Face detection dataset not found or incomplete")
    
    # Check emotion dataset
    emotion_train_dir = "datasets/Emotion/train"
    emotion_test_dir = "datasets/Emotion/test"
    
    emotion_dataset_available = os.path.exists(emotion_train_dir)
    
    if emotion_dataset_available:
        emotion_classes = [d for d in os.listdir(emotion_train_dir) 
                          if os.path.isdir(os.path.join(emotion_train_dir, d))]
        total_emotion_samples = 0
        for emotion_class in emotion_classes:
            class_dir = os.path.join(emotion_train_dir, emotion_class)
            class_samples = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            total_emotion_samples += class_samples
            logger.info(f"Emotion class '{emotion_class}': {class_samples} samples")
        
        logger.info(f"Total emotion samples: {total_emotion_samples}")
    else:
        logger.warning("Emotion dataset not found")
    
    return face_dataset_available, emotion_dataset_available

def train_face_detection(epochs=100, batch_size=16, learning_rate=0.001):
    """
    Train the face detection model
    """
    logger.info("="*50)
    logger.info("STARTING FACE DETECTION MODEL TRAINING")
    logger.info("="*50)
    
    start_time = time.time()
    
    try:
        # Initialize detector
        detector = FaceDetector()
        
        # Train model
        history = detector.train(
            images_dir="datasets/face detection/images/train",
            labels_dir="datasets/face detection/labels/train",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save model
        detector.save_model("models/face_detection_model.h5")
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        X_val, _, y_val, _ = detector.load_data(
            "datasets/face detection/images/val",
            "datasets/face detection/labels/val",
            validation_split=0.0
        )
        
        if len(X_val) > 0:
            val_loss = detector.model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"Validation Loss: {val_loss}")
        
        training_time = time.time() - start_time
        logger.info(f"Face detection training completed in {training_time/60:.2f} minutes")
        
        return detector, history
        
    except Exception as e:
        logger.error(f"Error during face detection training: {e}")
        raise

def train_emotion_recognition(epochs=150, batch_size=64, learning_rate=0.001):
    """
    Train the emotion recognition model
    """
    logger.info("="*50)
    logger.info("STARTING EMOTION RECOGNITION MODEL TRAINING")
    logger.info("="*50)
    
    start_time = time.time()
    
    try:
        # Initialize classifier
        classifier = EmotionClassifier()
        
        # Train model
        history = classifier.train(
            data_dir="datasets/Emotion/train",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_augmentation=True
        )
        
        # Save model
        classifier.save_model("models/emotion_detection_model.h5")
        
        # Evaluate on test set if available
        test_data_dir = "datasets/Emotion/test"
        if os.path.exists(test_data_dir):
            logger.info("Evaluating on test set...")
            X_test, _, y_test, _ = classifier.load_data(test_data_dir, validation_split=0.0)
            
            if len(X_test) > 0:
                test_results = classifier.model.evaluate(X_test, y_test, verbose=0)
                logger.info(f"Test Loss: {test_results[0]:.4f}")
                logger.info(f"Test Accuracy: {test_results[1]:.4f}")
                logger.info(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Emotion recognition training completed in {training_time/60:.2f} minutes")
        
        return classifier, history
        
    except Exception as e:
        logger.error(f"Error during emotion recognition training: {e}")
        raise

def create_model_config():
    """
    Create a configuration file for the trained models
    """
    config = {
        "face_detection": {
            "model_path": "models/face_detection_model.h5",
            "input_size": [224, 224],
            "confidence_threshold": 0.5
        },
        "emotion_recognition": {
            "model_path": "models/emotion_detection_model.h5",
            "input_size": [48, 48],
            "emotion_labels": ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        },
        "training_info": {
            "trained_on": datetime.now().isoformat(),
            "framework": "TensorFlow/Keras",
            "version": "1.0.0"
        }
    }
    
    import json
    os.makedirs('models', exist_ok=True)
    with open('models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Model configuration saved to models/model_config.json")

def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train Smart Presence ML Models')
    parser.add_argument('--face', action='store_true', help='Train face detection model')
    parser.add_argument('--emotion', action='store_true', help='Train emotion recognition model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--all', action='store_true', help='Train both models')
    
    args = parser.parse_args()
    
    # If no specific model is selected, train both
    if not args.face and not args.emotion and not args.all:
        args.all = True
    
    if args.all:
        args.face = True
        args.emotion = True
    
    logger.info("Starting Smart Presence Model Training")
    logger.info(f"Training configuration:")
    logger.info(f"  - Face Detection: {args.face}")
    logger.info(f"  - Emotion Recognition: {args.emotion}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Learning Rate: {args.learning_rate}")
    
    # Check datasets
    face_available, emotion_available = check_datasets()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    total_start_time = time.time()
    
    # Train face detection model
    if args.face:
        if not face_available:
            logger.error("Face detection dataset not available. Skipping face detection training.")
        else:
            try:
                face_detector, face_history = train_face_detection(
                    epochs=args.epochs,
                    batch_size=max(16, args.batch_size // 2),  # Smaller batch size for face detection
                    learning_rate=args.learning_rate
                )
                logger.info("Face detection model training completed successfully!")
            except Exception as e:
                logger.error(f"Face detection training failed: {e}")
    
    # Train emotion recognition model
    if args.emotion:
        if not emotion_available:
            logger.error("Emotion dataset not available. Skipping emotion recognition training.")
        else:
            try:
                emotion_classifier, emotion_history = train_emotion_recognition(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate
                )
                logger.info("Emotion recognition model training completed successfully!")
            except Exception as e:
                logger.error(f"Emotion recognition training failed: {e}")
    
    # Create model configuration
    create_model_config()
    
    total_training_time = time.time() - total_start_time
    logger.info("="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total training time: {total_training_time/60:.2f} minutes")
    logger.info("Models saved in 'models/' directory")
    logger.info("Training logs saved in 'training.log'")
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 