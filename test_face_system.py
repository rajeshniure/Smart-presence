#!/usr/bin/env python3
"""
Test script for face detection and recognition system
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartPresence.settings')
django.setup()

from attendance.utils.face_pipeline import test_pipeline, detect_and_recognize, load_models

def main():
    print("=== Face Detection and Recognition System Test ===\n")
    
    # Test 1: Check if models can be loaded
    print("1. Testing model loading...")
    models_loaded = load_models()
    if models_loaded:
        print("✅ Models loaded successfully")
    else:
        print("❌ Failed to load models")
        return False
    
    # Test 2: Test the pipeline
    print("\n2. Testing face detection and recognition pipeline...")
    pipeline_success = test_pipeline()
    if pipeline_success:
        print("✅ Pipeline test successful")
    else:
        print("❌ Pipeline test failed")
    
    # Test 3: Check for required files
    print("\n3. Checking required files...")
    required_files = [
        'facenet_svm.joblib',
        'facenet_label_encoder.joblib',
    ]
    
    # Check YOLO model
    yolo_paths = [
        'yolo_runs/face_yolo/weights/best.pt',
        'yolo_runs/face_yolo2/weights/best.pt',
        'yolo_runs/face_yolo/best.pt',
        'best.pt'
    ]
    
    yolo_found = False
    for path in yolo_paths:
        if os.path.exists(path):
            print(f"✅ YOLO model found: {path}")
            yolo_found = True
            break
    
    if not yolo_found:
        print("❌ YOLO model not found")
    
    # Check other files
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
    
    # Test 4: Check for test images
    print("\n4. Checking for test images...")
    test_dirs = [
        'datasets/face_recognition',
        'media/student_images',
        'static/img'
    ]
    
    test_image_found = False
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"✅ Test image found: {os.path.join(test_dir, file)}")
                    test_image_found = True
                    break
        if test_image_found:
            break
    
    if not test_image_found:
        print("❌ No test images found")
    
    print("\n=== Test Summary ===")
    if models_loaded and pipeline_success and yolo_found:
        print("✅ All tests passed! The system should work.")
    else:
        print("❌ Some tests failed. Check the issues above.")
    
    return models_loaded and pipeline_success and yolo_found

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 