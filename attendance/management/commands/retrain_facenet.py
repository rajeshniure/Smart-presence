from django.core.management.base import BaseCommand
from django.conf import settings
import os
import sys
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import logging

from attendance.models import Student

class Command(BaseCommand):
    help = 'Retrain FaceNet model using student photos from database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retraining even if models exist',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting FaceNet retraining from database...'))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Paths
        embeddings_path = 'facenet_embeddings.npy'
        labels_path = 'facenet_labels.npy'
        classifier_path = 'facenet_svm.joblib'
        encoder_path = 'facenet_label_encoder.joblib'
        
        # Check if models exist
        if not options['force']:
            existing_files = [embeddings_path, labels_path, classifier_path, encoder_path]
            if all(os.path.exists(f) for f in existing_files):
                self.stdout.write(
                    self.style.WARNING('Models already exist. Use --force to retrain.')
                )
                return
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stdout.write(f"Using device: {device}")
        
        # Load FaceNet model
        self.stdout.write("Loading FaceNet models...")
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        mtcnn = MTCNN(image_size=160, margin=0, device=device)
        
        # Get all students from database
        students = Student.objects.all()
        self.stdout.write(f"Found {students.count()} students in database")
        
        if students.count() == 0:
            self.stdout.write(
                self.style.ERROR('No students found in database. Please add students first.')
            )
            return
        
        # Prepare data
        embeddings = []
        labels = []
        processed_count = 0
        failed_count = 0
        
        for student in tqdm(students, desc='Processing students'):
            try:
                # Check if student has an image
                if not student.image:
                    self.stdout.write(
                        self.style.WARNING(f"Student {student.name} has no image, skipping...")
                    )
                    failed_count += 1
                    continue
                
                # Get the image path
                image_path = student.image.path
                
                # Check if file exists
                if not os.path.exists(image_path):
                    self.stdout.write(
                        self.style.WARNING(f"Image file not found for {student.name}: {image_path}")
                    )
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
                    self.stdout.write(f"Successfully processed {student.name}")
                else:
                    self.stdout.write(
                        self.style.WARNING(f"No face detected in image for {student.name}")
                    )
                    failed_count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error processing {student.name}: {str(e)}")
                )
                failed_count += 1
                continue
        
        if len(embeddings) == 0:
            self.stdout.write(
                self.style.ERROR('No valid embeddings generated. Cannot train model.')
            )
            return
        
        # Convert to numpy arrays
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        self.stdout.write(f"Generated {len(embeddings)} embeddings from {processed_count} students")
        self.stdout.write(f"Failed to process {failed_count} students")
        
        # Save embeddings and labels
        self.stdout.write("Saving embeddings and labels...")
        np.save(embeddings_path, embeddings)
        np.save(labels_path, labels)
        
        # Encode labels
        self.stdout.write("Encoding labels...")
        encoder = LabelEncoder()
        labels_num = encoder.fit_transform(labels)
        joblib.dump(encoder, encoder_path)
        
        # Train SVM classifier
        self.stdout.write("Training SVM classifier...")
        clf = SVC(kernel='linear', probability=True)
        clf.fit(embeddings, labels_num)
        joblib.dump(clf, classifier_path)
        
        self.stdout.write(
            self.style.SUCCESS('FaceNet retraining completed successfully!')
        )
        self.stdout.write(f"Model files saved:")
        self.stdout.write(f"  - Embeddings: {embeddings_path}")
        self.stdout.write(f"  - Labels: {labels_path}")
        self.stdout.write(f"  - Classifier: {classifier_path}")
        self.stdout.write(f"  - Encoder: {encoder_path}")
        
        # Print summary
        unique_labels = np.unique(labels)
        self.stdout.write(f"Model trained on {len(unique_labels)} unique students:")
        for label in unique_labels:
            count = np.sum(labels == label)
            self.stdout.write(f"  - {label}: {count} embeddings")
        
        self.stdout.write(
            self.style.SUCCESS('✅ FaceNet retraining completed successfully!')
        )
        self.stdout.write(
            self.style.WARNING('🔄 Please restart the Django server to load the new models.')
        ) 