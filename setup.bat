@echo off
REM Smart Presence Project Setup Script for Windows
REM This script sets up the project after cloning from git repository.

echo 🚀 Smart Presence Project Setup
echo ================================

REM Check if we're in the right directory
if not exist "manage.py" (
    echo ❌ manage.py not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Create necessary directories
echo.
echo 🔄 Creating necessary directories...
if not exist "media" mkdir media
if not exist "media\student_images" mkdir media\student_images
if not exist "datasets" mkdir datasets
if not exist "yolo_runs" mkdir yolo_runs
if not exist "yolo_runs\face_yolo" mkdir yolo_runs\face_yolo
if not exist "yolo_runs\face_yolo\weights" mkdir yolo_runs\face_yolo\weights
echo ✅ Directories created

REM Install Python requirements
echo.
echo 🔄 Installing Python requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python packages
    pause
    exit /b 1
)
echo ✅ Python packages installed

REM Setup database
echo.
echo 🔄 Setting up database...
python manage.py makemigrations
if %errorlevel% neq 0 (
    echo ❌ Failed to create database migrations
    pause
    exit /b 1
)
echo ✅ Database migrations created

python manage.py migrate
if %errorlevel% neq 0 (
    echo ❌ Failed to apply database migrations
    pause
    exit /b 1
)
echo ✅ Database migrations applied

REM Download YOLO weights if they don't exist
if not exist "yolov8n.pt" (
    echo.
    echo 🔄 Downloading YOLO weights...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt' -OutFile 'yolov8n.pt'"
    if exist "yolov8n.pt" (
        echo ✅ YOLO weights downloaded
    ) else (
        echo ⚠️ Failed to download YOLO weights. You may need to download manually.
    )
) else (
    echo ✅ YOLO weights already exist, skipping download
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo.
    echo 🔧 Creating .env file template...
    (
        echo # Django Settings
        echo DEBUG=True
        echo SECRET_KEY=your-secret-key-here-change-this-in-production
        echo.
        echo # Database Settings
        echo DATABASE_URL=sqlite:///db.sqlite3
        echo.
        echo # Media Settings
        echo MEDIA_URL=/media/
        echo MEDIA_ROOT=media/
        echo.
        echo # Face Recognition Settings
        echo FACE_RECOGNITION_THRESHOLD=0.6
    ) > .env
    echo ✅ .env file created
) else (
    echo ✅ .env file already exists
)

echo.
echo ====================================================================
echo 🎉 SETUP COMPLETED SUCCESSFULLY!
echo ====================================================================
echo.
echo 📋 Next steps to complete your setup:
echo.
echo 1. 🎭 Train Face Recognition Models:
echo    python manage.py train_simple_recognition
echo    python manage.py train_mtcnn
echo    python manage.py retrain_facenet
echo.
echo 2. 👥 Create Superuser ^(Admin^):
echo    python manage.py createsuperuser
echo.
echo 3. 🖼️ Upload Student Images:
echo    - Place student photos in media\student_images\
echo    - Or upload through Django admin interface
echo.
echo 4. 🚀 Run the Development Server:
echo    python manage.py runserver
echo.
echo 5. 🌐 Open your browser and go to:
echo    http://127.0.0.1:8000
echo.
echo ⚠️  Important Notes:
echo    - The .env file contains sensitive data, don't commit it to git
echo    - Training models may take several minutes depending on your hardware
echo    - Make sure you have sufficient disk space for models and datasets
echo.
echo ====================================================================
pause
