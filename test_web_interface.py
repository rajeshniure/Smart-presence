#!/usr/bin/env python3
"""
Test script for the web interface
"""

import requests
import time

def test_web_interface():
    """Test the web interface endpoints"""
    
    base_url = 'http://127.0.0.1:8000'
    
    print("=== Testing Web Interface ===\n")
    
    # Test 1: Check if scan page loads
    print("1. Testing scan page...")
    try:
        response = requests.get(f'{base_url}/scan/')
        if response.status_code == 200:
            print("✅ Scan page loads successfully")
        else:
            print(f"❌ Scan page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing scan page: {e}")
    
    # Test 2: Test API endpoint directly
    print("\n2. Testing API endpoint...")
    try:
        response = requests.get(f'{base_url}/api/test-pipeline/')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API test endpoint works: {data}")
        else:
            print(f"❌ API test endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing API: {e}")
    
    # Test 3: Test with a sample image
    print("\n3. Testing with sample image...")
    try:
        with open('media/student_images/001_webcam.jpeg', 'rb') as f:
            files = {'image': f}
            response = requests.post(f'{base_url}/api/scan/', files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Scan API works: {data}")
        else:
            print(f"❌ Scan API failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing scan API: {e}")
    
    print("\n=== Test Summary ===")
    print("If all tests pass, the web interface should work properly.")
    print("Try accessing http://127.0.0.1:8000/scan/ in your browser.")

if __name__ == "__main__":
    test_web_interface() 