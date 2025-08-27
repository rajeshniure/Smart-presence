import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class TrainingAnalytics:
    """
    Training analytics and visualization for simple face recognition system
    """
    
    def __init__(self):
        self.results = {}
        self.training_history = []
        
    def extract_features_from_image(self, image_path: str, face_cascade) -> np.ndarray:
        """Extract features from a single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                return None
                
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (64, 64))
            features = face_resized.flatten() / 255.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def prepare_training_data(self, student_data: List[Tuple[str, str]], face_cascade) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and labels"""
        features_list = []
        labels_list = []
        
        for student_name, image_path in student_data:
            features = self.extract_features_from_image(image_path, face_cascade)
            if features is not None:
                features_list.append(features)
                labels_list.append(student_name)
                logger.info(f"Processed {student_name}")
        
        return np.array(features_list), np.array(labels_list)
    
    def train_with_analytics(self, student_data: List[Tuple[str, str]], test_size=0.2, random_state=42):
        """Train the model with comprehensive analytics"""
        try:
            logger.info("Starting training with analytics...")
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades
            face_cascade_path = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            if face_cascade.empty():
                logger.error("Failed to load Haar cascade classifier")
                return False
            
            # Prepare data
            X, y = self.prepare_training_data(student_data, face_cascade)
            
            if len(X) == 0:
                logger.error("No valid features extracted")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            # Train model
            knn_classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
            knn_classifier.fit(X_train, y_train_encoded)
            
            # Predictions
            y_train_pred = knn_classifier.predict(X_train)
            y_test_pred = knn_classifier.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
            test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
            
            # Store results
            self.results = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'y_train_encoded': y_train_encoded,
                'y_test_encoded': y_test_encoded,
                'label_encoder': label_encoder,
                'classifier': knn_classifier,
                'student_data': student_data
            }
            
            logger.info(f"Training completed - Train Accuracy: {train_accuracy:.3f}, Test Accuracy: {test_accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error in training with analytics: {str(e)}")
            return False
    
    def generate_confusion_matrix(self, save_path='training_results/confusion_matrix.png'):
        """Generate and save confusion matrix"""
        try:
            if not self.results:
                logger.error("No training results available")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Generate confusion matrix
            cm = confusion_matrix(self.results['y_test_encoded'], self.results['y_test_pred'])
            
            # Get class names
            class_names = self.results['label_encoder'].classes_
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix - Face Recognition')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {str(e)}")
            return False
    
    def generate_accuracy_comparison(self, save_path='training_results/accuracy_comparison.png'):
        """Generate accuracy comparison chart"""
        try:
            if not self.results:
                logger.error("No training results available")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare data
            accuracies = [self.results['train_accuracy'], self.results['test_accuracy']]
            labels = ['Training Accuracy', 'Test Accuracy']
            colors = ['#2E8B57', '#4169E1']
            
            # Create plot
            plt.figure(figsize=(8, 6))
            bars = plt.bar(labels, accuracies, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Training vs Test Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.1)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Accuracy comparison saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating accuracy comparison: {str(e)}")
            return False
    
    def generate_class_performance(self, save_path='training_results/class_performance.png'):
        """Generate per-class performance analysis"""
        try:
            if not self.results:
                logger.error("No training results available")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Get classification report
            report = classification_report(
                self.results['y_test_encoded'], 
                self.results['y_test_pred'],
                target_names=self.results['label_encoder'].classes_,
                output_dict=True
            )
            
            # Extract per-class metrics
            class_names = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for class_name in self.results['label_encoder'].classes_:
                if class_name in report:
                    class_names.append(class_name)
                    precisions.append(report[class_name]['precision'])
                    recalls.append(report[class_name]['recall'])
                    f1_scores.append(report[class_name]['f1-score'])
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Precision and Recall
            x = np.arange(len(class_names))
            width = 0.35
            
            ax1.bar(x - width/2, precisions, width, label='Precision', alpha=0.7)
            ax1.bar(x + width/2, recalls, width, label='Recall', alpha=0.7)
            ax1.set_xlabel('Students')
            ax1.set_ylabel('Score')
            ax1.set_title('Precision and Recall by Student')
            ax1.set_xticks(x)
            ax1.set_xticklabels(class_names, rotation=45)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # F1 Scores
            ax2.bar(class_names, f1_scores, alpha=0.7, color='orange')
            ax2.set_xlabel('Students')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('F1 Score by Student')
            ax2.set_xticklabels(class_names, rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Class performance analysis saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating class performance: {str(e)}")
            return False
    
    def generate_training_summary(self, save_path='training_results/training_summary.txt'):
        """Generate comprehensive training summary"""
        try:
            if not self.results:
                logger.error("No training results available")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Calculate additional metrics
            total_samples = len(self.results['X_train']) + len(self.results['X_test'])
            unique_classes = len(self.results['label_encoder'].classes_)
            
            # Generate classification report
            report = classification_report(
                self.results['y_test_encoded'], 
                self.results['y_test_pred'],
                target_names=self.results['label_encoder'].classes_,
                output_dict=True
            )
            
            # Write summary
            with open(save_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("FACE RECOGNITION TRAINING SUMMARY\n")
                f.write("=" * 60 + "\n")
                f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Algorithm: K-Nearest Neighbors (KNN)\n")
                f.write(f"Feature Extraction: Pixel-based (64x64 flattened)\n")
                f.write(f"Face Detection: Haar Cascade\n\n")
                
                f.write("DATASET STATISTICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Samples: {total_samples}\n")
                f.write(f"Training Samples: {len(self.results['X_train'])}\n")
                f.write(f"Test Samples: {len(self.results['X_test'])}\n")
                f.write(f"Number of Classes: {unique_classes}\n")
                f.write(f"Feature Dimension: {self.results['X_train'].shape[1]}\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Training Accuracy: {self.results['train_accuracy']:.3f} ({self.results['train_accuracy']*100:.1f}%)\n")
                f.write(f"Test Accuracy: {self.results['test_accuracy']:.3f} ({self.results['test_accuracy']*100:.1f}%)\n")
                f.write(f"Overfitting: {'Yes' if self.results['train_accuracy'] - self.results['test_accuracy'] > 0.1 else 'No'}\n\n")
                
                f.write("PER-CLASS PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                for class_name in self.results['label_encoder'].classes_:
                    if class_name in report:
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {report[class_name]['precision']:.3f}\n")
                        f.write(f"  Recall: {report[class_name]['recall']:.3f}\n")
                        f.write(f"  F1-Score: {report[class_name]['f1-score']:.3f}\n")
                        f.write(f"  Support: {report[class_name]['support']}\n\n")
                
                f.write("MODEL CONFIGURATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"K-Neighbors: 3\n")
                f.write(f"Weights: distance\n")
                f.write(f"Face Size: 64x64 pixels\n")
                f.write(f"Feature Type: Normalized pixel values\n\n")
                
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                if self.results['test_accuracy'] < 0.7:
                    f.write("- Consider increasing training data\n")
                    f.write("- Try different feature extraction methods\n")
                    f.write("- Adjust KNN parameters\n")
                else:
                    f.write("- Model performance is acceptable\n")
                    f.write("- Consider fine-tuning for better accuracy\n")
                
                f.write("=" * 60 + "\n")
            
            logger.info(f"Training summary saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating training summary: {str(e)}")
            return False
    
    def generate_all_analytics(self, student_data: List[Tuple[str, str]]):
        """Generate all training analytics and visualizations"""
        try:
            logger.info("Generating comprehensive training analytics...")
            
            # Train model with analytics
            if not self.train_with_analytics(student_data):
                return False
            
            # Generate all visualizations
            self.generate_confusion_matrix()
            self.generate_accuracy_comparison()
            self.generate_class_performance()
            self.generate_training_summary()
            
            # Save trained model
            model_path = 'training_results/simple_face_classifier.pkl'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'classifier': self.results['classifier'],
                    'encoder': self.results['label_encoder']
                }, f)
            
            logger.info("All training analytics generated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error generating analytics: {str(e)}")
            return False

# Global instance
training_analytics = TrainingAnalytics() 