Build a Dynamic automated Smart Attendance and emotion detection web app using **Django**, **HTML**, **CSS**, **JavaScript**, **Tailwind CSS**, and **SQLite**. Focus only on the required working features below.

## Key Features to Implement

### 1. Face Recognition-Based Attendance (Live Webcam)
- **Face Detection**: Implement using MTCNN manually.

- **Face Recognition**: Implement using faceNet. use the pretrained facenet model for face recognition.

### 2. Sentiment Detection During Attendance
- Detect students' emotions (happy, sad, angry, neutral,disgusted, fearful, surprised) in real-time during attendance using a **manually coded CNN** (Convolutional Neural Network).


### 3. add the proper registation section  for registering the student which include
 -live webcam and image upload 
 -student details form

### 4. Time-Based Attendance Validation
- Check-in time 8:45 AM – 9:10 AM → Status: "Present"
- Check-in time 9:11 AM – 10:00 AM → Status: "Late"
- After 10:00 AM → Status: "Absent"
- **Check-out after 4:00 PM** is required to complete attendance.

### 5. Student ID Display on Recognition
- On recognizing a student, display a card-style UI showing:
  - Photo, Name, Student ID, Department
  - Attendance Status (Present/Late/Absent)
  - Sentiment (Happy, Sad, Angry, Neutral)

### 5. Keep the UI clean ,modern,minimilist, formated and properly aligned with proper layout and more dynamic and interactive :**
   - Use Tailwind and Css for modern style
   - modern and consistent colors, aligned inputs
   - use js for modern interctivity


### UI Structure Overview ###

## Universal Landing Page (Before Login)
**Path:** `/`

- **Navbar:**
  - Logo
  - Home | Login as Teacher | Login as Student
- **Main Content:**
  - Hero Section with System Introduction/overview
  - Features
  - Login Buttons/ quick actions
- **Footer:**
  - Contact Info
  - Privacy Policy, Terms of Service

---
## Teacher/Admin Dashboard (Only for Registered Teachers/Admins)
**Base Path:** `/teacher/`

### Shared Elements

- **Navbar:** Teacher/admin Home | Register Student | Students | Attendance Logs | Logout
-- **Main Content:** Dynamic based on selected page
- **Footer:**


### 1. Teacher Home Page
**Path:** `/teacher/`

- **System design and overview**
- **Statistics Cards**: Total students, today's present/late/absent counts
- **Quick Actions**: Register student, scan attendance, train model
- **Visual Indicators**: Color-coded status icons
- **Navigation**: Central hub for all system functions

### 2. Student Registration Page
**Path:** `/teacher/register/`

**Purpose**: Register new students with face images
**Features**:
- **Student Information Form**: Name, roll number, email, phone, department
- **Dual Image Upload**: File upload + webcam capture
- **Live Webcam Preview**: Real-time face capture
- **Validation**: Form validation with error handling

### 3. Student Management Page
**Path:** `/teacher/students/`

**Purpose**: View and manage all registered students
**Features**:
- **Student Directory**: Paginated list of all students
- **Student Cards**: Photo, name, roll number, department
- **Quick Actions**: Edit, delete, view details
- **Search & Filter**: Find students quickly
- **Bulk Operations**: Multiple student management

### 4. Attendance Logs Page
**Path:** `/teacher/attendance/`

**Purpose**: View and manage attendance records

**Features**:
- **Date Filtering**: View attendance by specific date
- **Status Overview**: Present/Late/Absent counts
- **Detailed Table**: Student Name, ID, Time In, Time Out, Status
- **Export Options**: Print-friendly view

## Student Dashboard (Only for Registered Students)
**Base Path:** `/student/`

### Shared Elements

- **Navbar:** Student Home | Scan | Attendance History | Logout
- **Main Content:** Dynamic based on selected page
- **Footer:**

### 1. Student Home Page
**Path:** `/student/`

- Welcome Message
- overview
- Personal Attendance Stats
- Attendance Streak Tracker
- quick navigation

### 2. Scan Page
**Path:** `/student/scan/`

**Purpose**: Live face recognition for attendance marking

**Features**:
- Live Webcam Feed
- **Face Recognition and Emotion detection**: Automatic face detection and recognition and Emotion Detection
- Student Card Display (Name, ID, Status, Sentiment)
- **Instant Feedback**
- **Attendance Logging**: Automatic attendance marking

### 3. Attendance History Page
**Path:** `/student/history/`

- Date Range Filter
- Attendance Table:
  - Date, Time In, Time Out, Status, Emotion Log
- Download/Export Option

---

## Access Control Flow Summary
| User Type | Dashboard Path | Access Restrictions                  |
| --------- | -------------- | ------------------------------------ |
| Teacher   | `/teacher/`    | Only for admin/teacher roles         |
| Student   | `/student/`    | Only registered students via teacher |

---
### Build a fully working Face Recognition Attendance System using the following tech stack:
- Backend: Django (Python), SQLite
- Frontend: HTML, CSS, JavaScript, Tailwind CSS

# properly place the HTML file in the templates folder and CSS,js under the static folder make the proper connection between templates and static folder

# Build the ui, frontend and backend only. dont add the model training and python machinelearning files and folder. i will add the algorithm and machine learning later.

# Dont add unnecessary file and folder make the code more modular and Dynamic.