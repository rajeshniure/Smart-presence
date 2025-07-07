

Build a simple Smart Attendance and Student Monitoring System web app using **Django**, **HTML**, **CSS**, **JavaScript**, **Bootstrap**, and **SQLite**. Focus only on the required working features below.

## Key Features to Implement

### 1. Face Recognition-Based Attendance (Live Webcam)
- **Face Detection**: Implement using MTCNN manually.
I have provided data set for face detection:
datasets
       ->face detection
                    ->images
                           train data
                           val data
                    -> labels
                            train data
                            val data

properly train the data for face detection using the given dataset.

- **Face Recognition**: Implement using faceNet. use the pretrained facenet model for face recognition.

### 2. Sentiment Detection During Attendance
- Detect students' emotions (happy, sad, angry, neutral,disgusted, fearful, surprised) in real-time during attendance using a **manually coded CNN** (Convolutional Neural Network).

the dataset for emotion detection is given as:
        datasets
          ->Emotion
             ->test data
             ->train data

properly train the data for emotion detection. use the hardcoded manually coded algorithm for emotion detection.

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

   - Use Bootstrap and Css for modern style
   - Minimal colors, aligned inputs
   - use js for modern interctivity


## UI Structure

- **Navbar:** Home, Register, Scan, Students, Attendence
- **Main Content:** Dynamic based on selected page

## Pages & Core Features

### 1. Home (`/`)
**Purpose**: Main landing page with system overview

**Features**:
- **Statistics Cards**: Total students, today's present/late/absent counts
- **Quick Actions**: Register student, scan attendance, train model
- **Visual Indicators**: Color-coded status icons
- **Navigation**: Central hub for all system functions



### 2. Student Registration (`/register/`)
**Purpose**: Register new students with face images

**Features**:
- **Student Information Form**: Name, roll number, email, phone, department
- **Dual Image Upload**: File upload + webcam capture
- **Live Webcam Preview**: Real-time face capture
- **Validation**: Form validation with error handling


### 3. Attendance Scanning (`/scan/`)
**Purpose**: Live face recognition for attendance marking

**Features**:
- **Live Video Feed**: Real-time webcam streaming
- **Face Recognition and Emotion detection**: Automatic face detection and recognition and Emotion Detection
- **Instant Feedback**
- **Attendance Logging**: Automatic attendance marking
- **Confidence Display**: Recognition confidence percentage


### 4. Student Management (`/students/`)
**Purpose**: View and manage all registered students

**Features**:
- **Student Directory**: Paginated list of all students
- **Student Cards**: Photo, name, roll number, department
- **Quick Actions**: Edit, delete, view details
- **Search & Filter**: Find students quickly
- **Bulk Operations**: Multiple student management


### 5. Attendance Logs (`/attendance/`)
**Purpose**: View and manage attendance records

**Features**:
- **Date Filtering**: View attendance by specific date
- **Status Overview**: Present/Late/Absent counts
- **Detailed Table**: Time stamps and status for each student
- **Export Options**: Print-friendly view
- **Auto-marking**: Automatic absent marking after cutoff time


### Build a fully working Face Recognition Attendance System using the following tech stack:

- Backend: Django (Python), SQLite
- Frontend: HTML, CSS, JavaScript, Bootstrap


# properly place the HTML file in the templates folder and CSS and js under the static folder make the proper connection between templates and static folder