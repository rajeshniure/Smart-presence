# SmartPresence - Quick Start (Windows)

A Django-based face recognition attendance system. This guide focuses on Windows only and keeps things simple for collaborators.

## ✅ Prerequisites (Windows)
- Python 3.8+ (add to PATH during install)
- Git for Windows
- Optional: NVIDIA GPU + CUDA for faster training

## 👥 Collaborate via GitHub
1. Fork the repository (or get access to the main repo).
2. Clone it:
   ```bash
   git clone <your-repository-url>
   cd Attendence
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/my-change
   ```
4. Make changes, then commit and push:
   ```bash
   git add .
   git commit -m "Describe your change"
   git push origin feature/my-change
   ```
5. Open a Pull Request on GitHub.

## ▶️ Run the Project (Windows)
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the database:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```
4. Create an admin user (follow prompts):
   ```bash
   python manage.py createsuperuser
   ```
5. Start the server:
   ```bash
   python manage.py runserver
   ```
6. Open the app at `http://127.0.0.1:8000/` and the admin at `http://127.0.0.1:8000/admin/`.

## 🧑‍🏫 Train Models (Windows)
Artifacts are saved to `attendance/models/`. Adjust `--dataset-path` if your data is elsewhere.

- Face detection (YOLO-style labels under `datasets/face detection/Images` + `labels`):
  ```bash
  python manage.py train_custom_detector --epochs 25 --batch-size 4 --learning-rate 0.0005 --dataset-path "datasets/face detection"
  ```

- Face recognition (folders per person under `datasets/face recognition/face_recognition_datasets`):
  ```bash
  py manage.py train_custom_recognition --epochs 30 --batch-size 32 --learning-rate 0.001 --dataset-path "datasets/face recognition/face_recognition_datasets"
  ```

- Classical baseline (LBP + SVM):
  ```bash
  python manage.py train_lbp_svm --dataset-path "datasets/face recognition/face_recognition_datasets" --test-size 0.2
  ```

- Generate training plots from history:
  ```bash
  python manage.py train_custom_pipeline
  ```

## 📂 Dataset Layout (expected)
- Detection (YOLO-style):
  ```
  datasets/face detection/
  ├── Images/
  │   ├── 0001.jpg
  │   └── ...
  └── labels/
      ├── 0001.txt   # lines like: 0 x_center y_center width height (normalized)
      └── ...
  ```
- Recognition (folders per identity):
  ```
  datasets/face recognition/
  └── face_recognition_datasets/
      ├── person_a/
      │   ├── 1.jpg
      │   └── ...
      └── person_b/
          ├── 1.jpg
          └── ...
  ```

## ❗ Tips (Windows)
- If Torch CUDA wheels fail, install CPU-only Torch or ensure CUDA is installed.
- Always activate the venv before running commands: `.venv\Scripts\activate`.
- After training, restart the server if models are reloaded at runtime.

