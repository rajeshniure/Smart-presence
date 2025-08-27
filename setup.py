#!/usr/bin/env python3
"""
Smart Presence Project Setup Script
This script sets up the project after cloning from git repository.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def download_file(url, filename, description):
    """Download a file from URL."""
    print(f"\n📥 Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'media',
        'media/student_images',
        'datasets',
        'yolo_runs',
        'yolo_runs/face_yolo',
        'yolo_runs/face_yolo/weights'
    ]
    
    print("\n📁 Creating necessary directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_database():
    """Set up the Django database."""
    print("\n🗄️ Setting up database...")
    
    # Check if manage.py exists
    if not os.path.exists('manage.py'):
        print("❌ manage.py not found. Make sure you're in the project root directory.")
        return False
    
    # Run Django commands
    commands = [
        ("python manage.py makemigrations", "Creating database migrations"),
        ("python manage.py migrate", "Applying database migrations"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    print("✅ Database setup completed")
    return True

def download_yolo_weights():
    """Download YOLO weights if they don't exist."""
    yolo_file = "yolov8n.pt"
    
    if os.path.exists(yolo_file):
        print(f"✅ {yolo_file} already exists, skipping download")
        return True
    
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    if download_file(yolo_url, yolo_file, "YOLO weights"):
        print("✅ YOLO weights downloaded successfully")
        return True
    else:
        print("⚠️ Failed to download YOLO weights. You may need to download manually.")
        return False

def install_requirements():
    """Install Python requirements."""
    print("\n📦 Installing Python requirements...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def create_env_file():
    """Create a basic .env file template."""
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ {env_file} already exists")
        return True
    
    print("\n🔧 Creating .env file template...")
    env_content = """# Django Settings
DEBUG=True
SECRET_KEY=your-secret-key-here-change-this-in-production

# Database Settings
DATABASE_URL=sqlite:///db.sqlite3

# Media Settings
MEDIA_URL=/media/
MEDIA_ROOT=media/

# Face Recognition Settings
FACE_RECOGNITION_THRESHOLD=0.6
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next steps to complete your setup:")
    print("\n1. 🎭 Train Face Recognition Models:")
    print("   python manage.py train_simple_recognition")
    print("   python manage.py train_mtcnn")
    print("   python manage.py retrain_facenet")
    print("\n2. 👥 Create Superuser (Admin):")
    print("   python manage.py createsuperuser")
    print("\n3. 🖼️ Upload Student Images:")
    print("   - Place student photos in media/student_images/")
    print("   - Or upload through Django admin interface")
    print("\n4. 🚀 Run the Development Server:")
    print("   python manage.py runserver")
    print("\n5. 🌐 Open your browser and go to:")
    print("   http://127.0.0.1:8000")
    print("\n⚠️  Important Notes:")
    print("   - The .env file contains sensitive data, don't commit it to git")
    print("   - Training models may take several minutes depending on your hardware")
    print("   - Make sure you have sufficient disk space for models and datasets")
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🚀 Smart Presence Project Setup")
    print("="*40)
    
    # Check if we're in the right directory
    if not os.path.exists('manage.py'):
        print("❌ Error: manage.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Setting up database", setup_database),
        ("Downloading YOLO weights", download_yolo_weights),
        ("Creating environment file", create_env_file),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        if not step_func():
            failed_steps.append(step_name)
    
    if failed_steps:
        print(f"\n❌ Setup failed for: {', '.join(failed_steps)}")
        print("Please check the errors above and try again.")
        sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()
