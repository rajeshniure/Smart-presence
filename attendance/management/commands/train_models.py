from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import sys
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from ml_models.face_detection import train_face_detection_model
from ml_models.emotion_detection import train_emotion_detection_model
from ml_models.inference import check_models_available

class Command(BaseCommand):
    help = 'Train face detection and emotion recognition models'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--face',
            action='store_true',
            help='Train face detection model',
        )
        parser.add_argument(
            '--emotion',
            action='store_true',
            help='Train emotion recognition model',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Train both models',
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='Number of training epochs (default: 50)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training (default: 32)',
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.001,
            help='Learning rate (default: 0.001)',
        )
        parser.add_argument(
            '--check-only',
            action='store_true',
            help='Only check if models and datasets are available',
        )
    
    def handle(self, *args, **options):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Check if we're only checking availability
        if options['check_only']:
            self.check_availability()
            return
        
        # If no specific model is selected, train both
        if not options['face'] and not options['emotion'] and not options['all']:
            options['all'] = True
        
        if options['all']:
            options['face'] = True
            options['emotion'] = True
        
        self.stdout.write(
            self.style.SUCCESS('Starting Smart Presence Model Training')
        )
        
        # Check datasets availability
        self.check_datasets()
        
        # Create models directory
        models_dir = os.path.join(settings.BASE_DIR, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Train face detection model
        if options['face']:
            self.stdout.write('Training face detection model...')
            try:
                self.train_face_detection(
                    epochs=options['epochs'],
                    batch_size=max(16, options['batch_size'] // 2),
                    learning_rate=options['learning_rate']
                )
                self.stdout.write(
                    self.style.SUCCESS('Face detection model trained successfully!')
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Face detection training failed: {e}')
                )
        
        # Train emotion recognition model
        if options['emotion']:
            self.stdout.write('Training emotion recognition model...')
            try:
                self.train_emotion_recognition(
                    epochs=options['epochs'],
                    batch_size=options['batch_size'],
                    learning_rate=options['learning_rate']
                )
                self.stdout.write(
                    self.style.SUCCESS('Emotion recognition model trained successfully!')
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Emotion recognition training failed: {e}')
                )
        
        self.stdout.write(
            self.style.SUCCESS('Model training completed!')
        )
    
    def check_availability(self):
        """Check if models and datasets are available"""
        self.stdout.write('Checking model and dataset availability...')
        
        # Check datasets
        face_dataset_available = self.check_face_dataset()
        emotion_dataset_available = self.check_emotion_dataset()
        
        # Check models
        models_status = check_models_available()
        
        self.stdout.write('\n' + '='*50)
        self.stdout.write('AVAILABILITY CHECK RESULTS')
        self.stdout.write('='*50)
        
        # Dataset status
        self.stdout.write('\nDatasets:')
        if face_dataset_available:
            self.stdout.write(self.style.SUCCESS('✓ Face detection dataset: Available'))
        else:
            self.stdout.write(self.style.ERROR('✗ Face detection dataset: Not found'))
        
        if emotion_dataset_available:
            self.stdout.write(self.style.SUCCESS('✓ Emotion dataset: Available'))
        else:
            self.stdout.write(self.style.ERROR('✗ Emotion dataset: Not found'))
        
        # Models status
        self.stdout.write('\nTrained Models:')
        if models_status['face_model_exists']:
            self.stdout.write(self.style.SUCCESS('✓ Face detection model: Available'))
        else:
            self.stdout.write(self.style.WARNING('⚠ Face detection model: Not found'))
        
        if models_status['emotion_model_exists']:
            self.stdout.write(self.style.SUCCESS('✓ Emotion recognition model: Available'))
        else:
            self.stdout.write(self.style.WARNING('⚠ Emotion recognition model: Not found'))
        
        if models_status['config_exists']:
            self.stdout.write(self.style.SUCCESS('✓ Model configuration: Available'))
        else:
            self.stdout.write(self.style.WARNING('⚠ Model configuration: Not found'))
        
        # Overall status
        self.stdout.write('\nOverall Status:')
        if models_status['all_models_available']:
            self.stdout.write(self.style.SUCCESS('✓ All models are ready for inference'))
        else:
            self.stdout.write(self.style.WARNING('⚠ Models need to be trained'))
        
        if face_dataset_available and emotion_dataset_available:
            self.stdout.write(self.style.SUCCESS('✓ All datasets are available for training'))
        else:
            self.stdout.write(self.style.ERROR('✗ Some datasets are missing'))
    
    def check_datasets(self):
        """Check if datasets are available"""
        face_available = self.check_face_dataset()
        emotion_available = self.check_emotion_dataset()
        
        if not face_available:
            self.stdout.write(
                self.style.WARNING('Face detection dataset not found')
            )
        
        if not emotion_available:
            self.stdout.write(
                self.style.WARNING('Emotion dataset not found')
            )
    
    def check_face_dataset(self):
        """Check if face detection dataset is available"""
        base_dir = settings.BASE_DIR
        
        required_dirs = [
            os.path.join(base_dir, "datasets/face detection/images/train"),
            os.path.join(base_dir, "datasets/face detection/labels/train"),
            os.path.join(base_dir, "datasets/face detection/images/val"),
            os.path.join(base_dir, "datasets/face detection/labels/val"),
        ]
        
        return all(os.path.exists(dir_path) for dir_path in required_dirs)
    
    def check_emotion_dataset(self):
        """Check if emotion dataset is available"""
        base_dir = settings.BASE_DIR
        emotion_train_dir = os.path.join(base_dir, "datasets/Emotion/train")
        
        if not os.path.exists(emotion_train_dir):
            return False
        
        # Check if emotion classes exist
        emotion_classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        for emotion in emotion_classes:
            emotion_dir = os.path.join(emotion_train_dir, emotion)
            if not os.path.exists(emotion_dir):
                return False
        
        return True
    
    def train_face_detection(self, epochs, batch_size, learning_rate):
        """Train face detection model"""
        from ml_models.face_detection import FaceDetector
        
        detector = FaceDetector()
        
        base_dir = settings.BASE_DIR
        train_images_dir = os.path.join(base_dir, "datasets/face detection/images/train")
        train_labels_dir = os.path.join(base_dir, "datasets/face detection/labels/train")
        
        history = detector.train(
            images_dir=train_images_dir,
            labels_dir=train_labels_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save model
        model_path = os.path.join(base_dir, "models/face_detection_model.h5")
        detector.save_model(model_path)
        
        return detector, history
    
    def train_emotion_recognition(self, epochs, batch_size, learning_rate):
        """Train emotion recognition model"""
        from ml_models.emotion_detection import EmotionClassifier
        
        classifier = EmotionClassifier()
        
        base_dir = settings.BASE_DIR
        train_data_dir = os.path.join(base_dir, "datasets/Emotion/train")
        
        history = classifier.train(
            data_dir=train_data_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_augmentation=True
        )
        
        # Save model
        model_path = os.path.join(base_dir, "models/emotion_detection_model.h5")
        classifier.save_model(model_path)
        
        return classifier, history 