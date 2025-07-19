import os
import numpy as np
import joblib

def check_model_students():
    """Check which students are currently in the trained model"""
    
    # Paths
    labels_path = 'facenet_labels.npy'
    encoder_path = 'facenet_label_encoder.joblib'
    
    print("🔍 Checking current model students...")
    
    # Check if model files exist
    if not os.path.exists(labels_path):
        print("❌ Model labels file not found. Model may not be trained.")
        return
    
    if not os.path.exists(encoder_path):
        print("❌ Model encoder file not found. Model may not be corrupted.")
        return
    
    try:
        # Load labels
        labels = np.load(labels_path)
        encoder = joblib.load(encoder_path)
        
        # Get unique students
        unique_students = np.unique(labels)
        
        print(f"✅ Model contains {len(unique_students)} students:")
        print("-" * 50)
        
        for i, student in enumerate(unique_students, 1):
            count = np.sum(labels == student)
            print(f"{i:2d}. {student} ({count} embeddings)")
        
        print("-" * 50)
        print(f"Total embeddings: {len(labels)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

if __name__ == '__main__':
    check_model_students() 