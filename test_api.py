#!/usr/bin/env python3
"""
Test script for the API endpoint
"""

import requests
import os

def test_api_with_image():
    """Test the API endpoint with a real image"""
    
    # Find a test image
    test_image_paths = [
        'media/student_images/001_webcam.jpeg',
        'media/student_images/1.jpg',
        'media/student_images/8.jpg'
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("❌ No test image found")
        return False
    
    print(f"Testing API with image: {test_image}")
    
    # Test the API endpoint
    url = 'http://127.0.0.1:8000/api/scan/'
    
    try:
        with open(test_image, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response: {data}")
            return True
        else:
            print(f"❌ API Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        return False

def test_pipeline_api():
    """Test the pipeline test endpoint"""
    url = 'http://127.0.0.1:8000/api/test-pipeline/'
    
    try:
        response = requests.get(url)
        print(f"Pipeline test status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pipeline test response: {data}")
            return True
        else:
            print(f"❌ Pipeline test error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== API Testing ===\n")
    
    print("1. Testing pipeline endpoint...")
    test_pipeline_api()
    
    print("\n2. Testing scan endpoint with image...")
    test_api_with_image() 