# Smart Presence - AI-Powered Attendance Management System

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Implementation](#machine-learning-implementation)
- [Database Design](#database-design)
- [Installation & Setup](#installation--setup)
- [Training the ML Models](#training-the-ml-models)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Project Overview

**Smart Presence** is an intelligent attendance management system that leverages cutting-edge AI and computer vision technologies to automate student attendance tracking. The system uses face recognition and detection to provide a seamless, contactless attendance experience while maintaining high accuracy and security.

### Key Highlights
- **AI-Powered Face Recognition**: Uses YOLOv8 for face detection and FaceNet for face recognition
- **Real-time Processing**: Instant attendance marking with live camera feed
- **Dual Mode Operation**: Check-in and Check-out functionality
- **Multi-User Support**: Separate interfaces for teachers and students
- **Comprehensive Analytics**: Detailed attendance reports and statistics
- **Timezone Aware**: Proper handling of Nepal timezone for accurate timestamps

## ✨ Features

### 🔐 Authentication & User Management
- **Role-based Access**: Separate interfaces for teachers (admin) and students
- **Secure Login**: Django authentication system
- **User Profile Management**: Students can update their profiles and photos
- **Automatic User Creation**: Creates Django user accounts when students are registered

### 📸 Face Recognition System
- **YOLOv8 Face Detection**: High-accuracy face detection using YOLOv8
- **FaceNet Recognition**: Deep learning-based face recognition
- **SVM Classification**: Support Vector Machine for person identification
- **Real-time Processing**: Live camera feed with instant recognition
- **Confidence Scoring**: Reliability metrics for recognition accuracy

### ⏰ Attendance Management
- **Check-in/Check-out System**: Dual functionality for entry and exit tracking
- **Time-based Status**: Automatic classification (Present/Late/Absent) based on check-in time
- **Configurable Thresholds**: Customizable time thresholds for attendance status
- **Timezone Support**: Nepal timezone (Asia/Kathmandu) integration
- **Duplicate Prevention**: Prevents multiple check-ins on the same day

### 📊 Analytics & Reporting
- **Real-time Statistics**: Live dashboard with attendance metrics
- **Detailed Reports**: Comprehensive attendance reports by date, student, or course
- **Export Functionality**: Downloadable reports in various formats
- **Visual Analytics**: Charts and graphs for attendance trends

### 🎨 User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, intuitive interface with Bootstrap styling
- **Real-time Feedback**: Live status updates and notifications
- **Accessibility**: User-friendly design with clear visual indicators

## 🏗️ System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   ML Pipeline   │
│   (HTML/CSS/JS) │◄──►│   (Django)      │◄──►│   (YOLO/FaceNet)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Feed   │    │   Database      │    │   Model Files   │
│   (WebRTC)      │    │   (SQLite)      │    │   (Trained)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **Camera Capture**: WebRTC captures live video feed
2. **Image Processing**: Frame extraction and preprocessing
3. **Face Detection**: YOLOv8 detects faces in the frame
4. **Face Recognition**: FaceNet extracts embeddings and SVM classifies
5. **Attendance Marking**: Django backend processes and stores attendance
6. **Response**: Real-time feedback to the user interface

## 🛠️ Technology Stack

### Backend Framework
- **Django 5.2.4**: Web framework for rapid development
- **Django REST Framework**: API development
- **SQLite**: Database (can be upgraded to PostgreSQL/MySQL)

### Machine Learning & Computer Vision
- **YOLOv8**: Real-time object detection for face detection
- **FaceNet**: Deep learning model for face recognition
- **MTCNN**: Multi-task Cascaded Convolutional Networks for face alignment
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **scikit-learn**: Machine learning utilities (SVM classifier)

### Frontend Technologies
- **HTML5/CSS3**: Structure and styling
- **JavaScript (ES6+)**: Client-side functionality
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icon library
- **WebRTC**: Camera access and video streaming

### Development Tools
- **Python 3.8+**: Programming language
- **Pillow**: Image processing
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **tqdm**: Progress bars

## 🤖 Machine Learning Implementation

### 1. Face Detection Pipeline (YOLOv8)

#### Training Process
```python
# Training Configuration
- Model: YOLOv8n (nano version for speed)
- Dataset: Custom face detection dataset
- Epochs: 20
- Image Size: 224x224
- Batch Size: 32
- Classes: 1 (face)
```

#### Dataset Structure
```
datasets/
└── face detection/
    └── images/
        ├── train/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── val/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
```

#### Training Script
```bash
python train_yolo.py
```

### 2. Face Recognition Pipeline (FaceNet + SVM)

#### Model Architecture
- **FaceNet (InceptionResnetV1)**: Pre-trained on VGGFace2 dataset
- **MTCNN**: Face detection and alignment
- **SVM Classifier**: Linear SVM for person identification
- **Label Encoder**: Converts names to numerical labels

#### Training Process
```python
# Training Steps
1. Load FaceNet model (pre-trained)
2. Process student images through MTCNN
3. Extract 512-dimensional embeddings
4. Train SVM classifier on embeddings
5. Save model files for inference
```

#### Dataset Structure
```
datasets/
└── face_recognition/
    ├── student1/
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── ...
    ├── student2/
    │   ├── photo1.jpg
    │   └── ...
    └── ...
```

#### Training Script
```bash
python train_facenet_recognition.py
```

### 3. Model Files Generated
- `yolo_runs/face_yolo/weights/best.pt`: Trained YOLOv8 model
- `facenet_embeddings.npy`: Face embeddings
- `facenet_labels.npy`: Corresponding labels
- `facenet_svm.joblib`: Trained SVM classifier
- `facenet_label_encoder.joblib`: Label encoder

## 🗄️ Database Design

### Core Models

#### Student Model
```python
class Student(models.Model):
    user = models.OneToOneField(User)  # Django user account
    name = models.CharField(max_length=100)
    roll_number = models.CharField(max_length=20, unique=True)
    email = models.EmailField(unique=True)
    department = models.CharField(max_length=100)
    image = models.ImageField(upload_to='student_images/')
    # Additional fields for comprehensive student info
```

#### Attendance Model
```python
class Attendance(models.Model):
    student = models.ForeignKey(Student)
    date = models.DateField()
    check_in_time = models.TimeField()
    check_out_time = models.TimeField()
    status = models.CharField(choices=[('present', 'Late', 'absent')])
    confidence = models.FloatField()  # Recognition confidence
```

#### Department & Course Models
```python
class Department(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10)

class Course(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=20)
    department = models.ForeignKey(Department)
    instructor = models.CharField(max_length=100)
```

#### Settings Model
```python
class AttendanceSettings(models.Model):
    on_time_threshold = models.TimeField()  # e.g., 9:10 AM
    late_threshold = models.TimeField()     # e.g., 10:00 AM
    absent_threshold = models.TimeField()   # e.g., 10:30 AM
    face_recognition_threshold = models.FloatField()  # e.g., 0.6
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git
- Webcam (for testing)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Smart-Presence
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 5: Create Superuser
```bash
python manage.py createsuperuser
```

### Step 6: Collect Static Files
```bash
python manage.py collectstatic
```

### Step 7: Run the Development Server
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## 🎓 Training the ML Models

### Step 1: Prepare Face Detection Dataset
1. Create the directory structure:
```bash
mkdir -p datasets/face\ detection/images/{train,val}
```

2. Add face images to train and validation folders
3. Create annotations in YOLO format (x_center, y_center, width, height)

### Step 2: Train YOLOv8 Model
```bash
python train_yolo.py
```

### Step 3: Prepare Face Recognition Dataset
1. Create the directory structure:
```bash
mkdir -p datasets/face_recognition
```

2. Add student photos:
```bash
datasets/face_recognition/
├── student1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
├── student2/
│   ├── photo1.jpg
│   └── ...
└── ...
```

### Step 4: Train FaceNet Model
```bash
python train_facenet_recognition.py
```

### Step 5: Verify Model Loading
```bash
python manage.py runserver
# Check console logs for model loading status
```

## 📖 Usage Guide

### For Teachers/Administrators

#### 1. Access Admin Panel
- Navigate to `http://127.0.0.1:8000/admin/`
- Login with superuser credentials

#### 2. Register Students
- Go to "Students" section
- Click "Add Student"
- Fill in student details and upload photo
- System automatically creates Django user account

#### 3. Configure Settings
- Go to "Attendance Settings"
- Set time thresholds for attendance status
- Configure face recognition confidence threshold

#### 4. Monitor Attendance
- Access teacher dashboard at `http://127.0.0.1:8000/teacher/`
- View real-time attendance statistics
- Generate reports by date, student, or course

### For Students

#### 1. Student Login
- Navigate to `http://127.0.0.1:8000/student/`
- Login with credentials provided by admin

#### 2. Check-in/Check-out
- Go to scan page: `http://127.0.0.1:8000/scan/`
- Click "Check-in" or "Check-out" button
- Look at the camera when prompted
- Wait for confirmation message

#### 3. View Attendance
- Access student dashboard
- View personal attendance history
- Check attendance statistics

### Check-in/Check-out Workflow

#### Check-in Process
1. **Click "Check-in" button**
2. **Camera activates** and requests permission
3. **Face detection** using YOLOv8
4. **Face recognition** using FaceNet + SVM
5. **Attendance marking** with current time
6. **Status determination** (Present/Late/Absent)
7. **Confirmation display** with student details

#### Check-out Process
1. **Click "Check-out" button**
2. **Camera activates** and requests permission
3. **Face detection** using YOLOv8
4. **Face recognition** using FaceNet + SVM
5. **Attendance update** with check-out time
6. **Confirmation display** with timing details

## 🔌 API Documentation

### Authentication
All APIs are currently open (no authentication required for demo purposes)

### Endpoints

#### 1. Scan API (Legacy)
```
POST /api/scan/
Content-Type: multipart/form-data
Body: image file
Response: JSON with recognition results
```

#### 2. Check-in API
```
POST /api/checkin/
Content-Type: multipart/form-data
Body: image file
Response: JSON with check-in results
```

#### 3. Check-out API
```
POST /api/checkout/
Content-Type: multipart/form-data
Body: image file
Response: JSON with check-out results
```

#### 4. Student Management APIs
```
GET /api/students/<id>/ - Get student details
PUT /api/students/<id>/ - Update student
DELETE /api/students/<id>/ - Delete student
```

### Response Format
```json
{
  "results": [
    {
      "name": "John Doe",
      "roll_number": "2021001",
      "department": "Computer Science",
      "attendance_status": "checked_in",
      "attendance_details": {
        "status": "present",
        "check_in_time": "09:15",
        "check_out_time": null
      },
      "photo_url": "/media/student_images/john.jpg"
    }
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
**Problem**: "YOLO model not found" or "SVM classifier not found"
**Solution**: 
- Ensure models are trained: `python train_yolo.py` and `python train_facenet_recognition.py`
- Check file paths in `attendance/utils/face_pipeline.py`

#### 2. Camera Access Issues
**Problem**: "Camera access failed"
**Solution**:
- Ensure browser has camera permissions
- Check if camera is being used by another application
- Try refreshing the page

#### 3. Face Recognition Accuracy
**Problem**: Low recognition accuracy
**Solution**:
- Improve training dataset quality
- Add more photos per student
- Adjust confidence threshold in settings
- Ensure good lighting conditions

#### 4. Timezone Issues
**Problem**: Incorrect timestamps
**Solution**:
- Verify `TIME_ZONE = 'Asia/Kathmandu'` in settings.py
- Ensure `USE_TZ = True` is set

#### 5. Database Migration Issues
**Problem**: Migration errors
**Solution**:
```bash
python manage.py makemigrations --merge
python manage.py migrate
```

### Performance Optimization

#### 1. Model Optimization
- Use GPU acceleration if available
- Consider model quantization for faster inference
- Optimize image preprocessing pipeline

#### 2. Database Optimization
- Add database indexes for frequently queried fields
- Implement caching for static data
- Use database connection pooling

#### 3. Frontend Optimization
- Implement lazy loading for images
- Use CDN for static assets
- Optimize JavaScript bundle size

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Commit your changes: `git commit -m 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions
- Include type hints where appropriate

### Testing
```bash
# Run tests
python manage.py test

# Run with coverage
coverage run --source='.' manage.py test
coverage report
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the face detection model
- **FaceNet**: Google for the face recognition architecture
- **Django**: Django Software Foundation for the web framework
- **OpenCV**: OpenCV team for computer vision tools
- **PyTorch**: Facebook AI Research for the deep learning framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the troubleshooting section above

---

**Smart Presence** - Making attendance management intelligent and efficient! 🚀 