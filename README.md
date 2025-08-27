# Smart Presence - Setup Guide

## 📦 Installation & Setup

### Option 1: Automated Setup (Recommended)

#### For Windows Users:
```bash
# Clone the repository
git clone <your-repository-url>
cd Smart-Presence

# Run the Windows setup script
setup.bat

```
#### For All Platforms (Python):
```bash
# Clone the repository
git clone <your-repository-url>
cd Smart-Presence

# Run the Python setup script
python setup.py
```

### Option 2: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd Smart-Presence
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the database:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **Download YOLO weights:**
   ```bash
   # Download manually from:
   # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   # Or use the setup scripts above
   ```

5. **Create necessary directories:**
   ```bash
   mkdir -p media/student_images
   mkdir -p datasets
   mkdir -p yolo_runs/face_yolo/weights
   ```

## 🎭 Training Face Recognition Models

After the initial setup, you need to train the face recognition models:

```bash
# Train simple recognition model
python manage.py train_simple_recognition

# Train MTCNN model
python manage.py train_mtcnn

# Retrain FaceNet model
python manage.py retrain_facenet
```

**Note:** Training may take several minutes depending on your hardware and dataset size.

## 👥 Initial Configuration

1. **Create a superuser (admin account):**
   ```bash
   python manage.py createsuperuser
   ```

2. **Upload student images:**
   - Place student photos in `media/student_images/` folder
   - Or upload through the Django admin interface at `/admin/`

3. **Configure attendance settings:**
   - Access the admin panel
   - Set up courses, departments, and attendance parameters

## 🚀 Running the Application

1. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

2. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:8000
   ```

3. **Access the admin panel:**
   ```
   http://127.0.0.1:8000/admin/
   ```
