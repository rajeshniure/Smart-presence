from django.core.management.base import BaseCommand
from django.conf import settings
import os
import logging

from attendance.models import Student
from attendance.utils.face_pipeline import train_simple_classifier, reload_models
from attendance.utils.training_analytics import training_analytics

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train simple face recognition system using student photos from database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retraining even if models exist',
        )
        parser.add_argument(
            '--analytics',
            action='store_true',
            help='Generate comprehensive training analytics and visualizations',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting simple face recognition training...'))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Check if models exist
        if not options['force']:
            if os.path.exists('simple_face_classifier.pkl'):
                self.stdout.write(
                    self.style.WARNING('Simple recognition model already exists. Use --force to retrain.')
                )
                return
        
        # Get all verified students from database
        students = Student.objects.filter(is_verified=True)
        self.stdout.write(f"Found {students.count()} verified students in database")
        
        if students.count() == 0:
            self.stdout.write(
                self.style.ERROR('No verified students found in database. Please add and verify students first.')
            )
            return
        
        # Prepare training data
        student_data = []
        processed_count = 0
        failed_count = 0
        
        for student in students:
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
                
                # Add to training data
                student_data.append((student.name, image_path))
                processed_count += 1
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error processing {student.name}: {str(e)}")
                )
                failed_count += 1
                continue
        
        if len(student_data) == 0:
            self.stdout.write(
                self.style.ERROR('No valid student data found. Cannot train model.')
            )
            return
        
        self.stdout.write(f"Prepared {len(student_data)} students for training")
        self.stdout.write(f"Failed to process {failed_count} students")
        
        # Train the model with or without analytics
        if options['analytics']:
            self.stdout.write("Training with comprehensive analytics...")
            success = training_analytics.generate_all_analytics(student_data)
            
            if success:
                self.stdout.write(
                    self.style.SUCCESS('Training with analytics completed successfully!')
                )
                self.stdout.write("Generated files:")
                self.stdout.write("  - training_results/confusion_matrix.png")
                self.stdout.write("  - training_results/accuracy_comparison.png")
                self.stdout.write("  - training_results/class_performance.png")
                self.stdout.write("  - training_results/training_summary.txt")
                self.stdout.write("  - training_results/simple_face_classifier.pkl")
                
                # Reload models to use the new classifier
                reload_models()
            else:
                self.stdout.write(
                    self.style.ERROR('Training with analytics failed!')
                )
                return
        else:
            # Standard training without analytics
            self.stdout.write("Training simple face recognition model...")
            success = train_simple_classifier(student_data)
            
            if success:
                # Reload models to use the new classifier
                reload_models()
                
                self.stdout.write(
                    self.style.SUCCESS('Simple face recognition training completed successfully!')
                )
                self.stdout.write(f"Model file saved: simple_face_classifier.pkl")
            else:
                self.stdout.write(
                    self.style.ERROR('Simple face recognition training failed!')
                )
                return
        
        # Print summary
        unique_names = list(set([name for name, _ in student_data]))
        self.stdout.write(f"Model trained on {len(unique_names)} unique students:")
        for name in sorted(unique_names):
            self.stdout.write(f"  - {name}")