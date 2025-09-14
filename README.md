

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



Simply run the script:
```bash

python face_recognition_plots.py
python face_detection_plots.py

# Run combined evaluation
python evaluation_plots.py

```
