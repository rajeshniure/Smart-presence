import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')

class FaceRecognitionPlotter:
    def __init__(self):
        self.recognition_history = None
        
    def load_training_history(self, history_path):
        """Load training history from pickle file"""
        try:
            with open(history_path, 'rb') as f:
                self.recognition_history = pickle.load(f)
            print("Face recognition training history loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading recognition history: {e}")
            return False
    
    def plot_training_metrics(self, model_name="Face Recognition"):
        """Plot training and validation metrics"""
        if self.recognition_history is None:
            print("No training history available!")
            return
        
        print(f" Generating {model_name} training metrics plots...")
        
        # Extract data
        epochs = range(1, len(self.recognition_history['train_losses']) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f'{model_name} - Training and Validation Metrics', fontsize=12, fontweight='bold')
        
        # Plot Loss
        axes[0, 0].plot(epochs, self.recognition_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.recognition_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Accuracy
        axes[0, 1].plot(epochs, self.recognition_history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.recognition_history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot Precision
        axes[0, 2].plot(epochs, self.recognition_history['val_precisions'], 'r-', label='Validation Precision', linewidth=2)
        axes[0, 2].set_title('Model Precision', fontweight='bold')
        axes[0, 2].set_xlabel('Epochs')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot Recall
        axes[1, 0].plot(epochs, self.recognition_history['val_recalls'], 'r-', label='Validation Recall', linewidth=2)
        axes[1, 0].set_title('Model Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot F1 Score
        axes[1, 1].plot(epochs, self.recognition_history['val_f1s'], 'r-', label='Validation F1-Score', linewidth=2)
        axes[1, 1].set_title('Model F1-Score', fontweight='bold')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Show final metrics summary
        final_metrics = {
            'Final Training Loss': self.recognition_history['train_losses'][-1],
            'Final Validation Loss': self.recognition_history['val_losses'][-1],
            'Best Validation Accuracy': self.recognition_history.get('best_val_acc', 0),
            'Final Training Accuracy': self.recognition_history['train_accuracies'][-1],
            'Final Validation Accuracy': self.recognition_history['val_accuracies'][-1],
            'Final Validation Precision': self.recognition_history['val_precisions'][-1],
            'Final Validation Recall': self.recognition_history['val_recalls'][-1],
            'Final Validation F1': self.recognition_history['val_f1s'][-1],
            'Total Epochs': len(self.recognition_history['train_losses']),
            'Number of Classes': len(self.recognition_history['class_names'])
        }
        
        metrics_text = '\n'.join([f'{k}: {v:.4f}' if isinstance(v, (int, float)) else f'{k}: {v}' for k, v in final_metrics.items()])
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Final Metrics Summary', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        print(" Displaying custom history plot...")
        plt.show()
        print(" Custom history plot displayed successfully!")
    
    def plot_confusion_matrix(self, model_name="Face Recognition"):
        """Plot confusion matrix from validation data"""
        if self.recognition_history is None:
            print("No training history available for confusion matrix!")
            return
        
        print(f" Generating {model_name} confusion matrix...")
        
        # Get the last confusion matrix (from final epoch)
        confusion_matrix = self.recognition_history['val_confusion_matrices'][-1]
        print(f"Confusion matrix shape: {confusion_matrix.shape}")
        
        # Get class names
        class_names = self.recognition_history.get('class_names', [])
        print(f"Number of classes: {len(class_names)}")
        
        # For large matrices, show 10 random classes
        if confusion_matrix.shape[0] > 50:
            print(" Large confusion matrix detected. Showing 10 random classes...")
            
            # Select 10 random classes
            np.random.seed(42)
            n_classes = min(10, confusion_matrix.shape[0])
            selected_indices = np.random.choice(confusion_matrix.shape[0], n_classes, replace=False)
            selected_indices = np.sort(selected_indices)
            
            # Create submatrix
            sub_confusion_matrix = confusion_matrix[np.ix_(selected_indices, selected_indices)]
            sub_class_names = [class_names[i] for i in selected_indices]
            
            print(f"Selected {n_classes} random classes: {selected_indices}")
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.imshow(sub_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{model_name} - Confusion Matrix (10 Random Classes)', fontsize=12, fontweight='bold')
            plt.colorbar()
            
            # Set labels
            tick_marks = np.arange(n_classes)
            plt.xticks(tick_marks, sub_class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, sub_class_names)
            
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            # Add text annotations
            thresh = sub_confusion_matrix.max() / 2.
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(j, i, format(sub_confusion_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if sub_confusion_matrix[i, j] > thresh else "black",
                            fontsize=12)
            
            # Add statistics
            total_samples = np.sum(confusion_matrix)
            correct_predictions = np.sum(np.diag(confusion_matrix))
            overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            plt.figtext(0.02, 0.02, f'Overall Accuracy: {overall_accuracy:.4f}\nTotal Samples: {total_samples}\nCorrect: {correct_predictions}\nShowing 10 random classes out of {confusion_matrix.shape[0]}', 
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        else:
            # For smaller matrices, show full detail
            plt.figure(figsize=(12, 10))
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{model_name} - Confusion Matrix (Final Epoch)', fontsize=16, fontweight='bold')
            plt.colorbar()
            
            # Set labels
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, class_names)
            
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            # Add text annotations
            thresh = confusion_matrix.max() / 2.
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if confusion_matrix[i, j] > thresh else "black",
                            fontsize=8)
        
        plt.tight_layout()
        print(" Displaying confusion matrix...")
        plt.show()
        print(" Confusion matrix displayed successfully!")
    
    def generate_evaluation_report(self):
        """Generate evaluation report"""
        if self.recognition_history is None:
            print("No recognition history available for report!")
            return
            
        print("="*60)
        print("FACE RECOGNITION MODEL EVALUATION REPORT")
        print("="*60)
        
        final_epoch = len(self.recognition_history['train_losses'])
        print(f"Total Epochs Trained: {final_epoch}")
        print(f"Final Training Loss: {self.recognition_history['train_losses'][-1]:.4f}")
        print(f"Final Validation Loss: {self.recognition_history['val_losses'][-1]:.4f}")
        print(f"Best Validation Accuracy: {self.recognition_history['best_val_acc']:.4f}")
        print(f"Final Training Accuracy: {self.recognition_history['train_accuracies'][-1]:.4f}")
        print(f"Final Validation Accuracy: {self.recognition_history['val_accuracies'][-1]:.4f}")
        print(f"Final Validation Precision: {self.recognition_history['val_precisions'][-1]:.4f}")
        print(f"Final Validation Recall: {self.recognition_history['val_recalls'][-1]:.4f}")
        print(f"Final Validation F1-Score: {self.recognition_history['val_f1s'][-1]:.4f}")
        print(f"Number of Classes: {len(self.recognition_history['class_names'])}")
        print("="*60)

def main():
    """Main function to run face recognition evaluation"""
    plotter = FaceRecognitionPlotter()
    
    # Load face recognition training history
    recognition_history_path = "face_recognition/recognition_training_history.pkl"
    if os.path.exists(recognition_history_path):
        plotter.load_training_history(recognition_history_path)
        
        # Generate plots and reports
        print("\n Generating Face Recognition Model Plots...")
        plotter.plot_training_metrics("Face Recognition")
        plotter.plot_confusion_matrix("Face Recognition")
        plotter.generate_evaluation_report()
        
        print("\n Face Recognition evaluation complete!")
    else:
        print(f"Warning: {recognition_history_path} not found!")

if __name__ == "__main__":
    main()