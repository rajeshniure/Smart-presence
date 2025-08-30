from django.core.management.base import BaseCommand
from django.conf import settings
import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from attendance.utils.dataset_loader import FaceRecognitionDataset


class Command(BaseCommand):
	help = "Train LBP+SVM face recognition model"

	def add_arguments(self, parser):
		parser.add_argument('--dataset-path', type=str, default='datasets/face recognition/face_recognition_datasets')
		parser.add_argument('--test-size', type=float, default=0.2)

	def handle(self, *args, **options):
		dataset_path = options['dataset_path']
		test_size = options['test_size']

		if not os.path.isdir(dataset_path):
			self.stdout.write(self.style.ERROR(f"Dataset path not found: {dataset_path}"))
			return

		dataset = FaceRecognitionDataset(dataset_path)
		class_names = dataset.get_class_names()

		self.stdout.write(f"Loaded {len(dataset)} images across {len(class_names)} classes")

		X, y = self._extract_lbp_features(dataset)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

		scaler = StandardScaler()
		X_train_s = scaler.fit_transform(X_train)
		X_test_s = scaler.transform(X_test)

		svm = SVC(kernel='rbf', probability=True)
		svm.fit(X_train_s, y_train)

		y_pred = svm.predict(X_test_s)
		acc = accuracy_score(y_test, y_pred)
		report = classification_report(y_test, y_pred, target_names=class_names)
		cm = confusion_matrix(y_test, y_pred)

		self.stdout.write(self.style.SUCCESS(f"Accuracy: {acc:.4f}"))
		self.stdout.write(report)
		self._save_confusion_matrix(cm, class_names)

		self._save_model(svm, scaler, class_names)

	def _extract_lbp_features(self, dataset):
		X = []
		y = []
		for tensor, label in dataset:
			img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			lbp = self._compute_lbp(gray)
			n_bins = 26  # for radius=3 uniform approx
			hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
			X.append(hist)
			y.append(int(label))
		return np.array(X), np.array(y)

	def _compute_lbp(self, gray):
		radius = 3
		n_points = 8 * radius
		h, w = gray.shape
		lbp = np.zeros_like(gray, dtype=np.uint8)
		for i in range(radius, h - radius):
			for j in range(radius, w - radius):
				c = gray[i, j]
				code = 0
				for k in range(n_points):
					angle = 2 * np.pi * k / n_points
					x = int(i + radius * np.cos(angle))
					y = int(j + radius * np.sin(angle))
					if gray[x, y] >= c:
						code |= (1 << k)
				lbp[i, j] = code
		return lbp

	def _save_model(self, svm, scaler, class_names):
		models_dir = os.path.join(settings.BASE_DIR, 'attendance', 'models')
		os.makedirs(models_dir, exist_ok=True)
		path = os.path.join(models_dir, 'lbp_svm_model.pkl')
		with open(path, 'wb') as f:
			pickle.dump({'svm': svm, 'scaler': scaler, 'class_names': class_names}, f)
		self.stdout.write(f"Saved model -> {path}")

	def _save_confusion_matrix(self, cm, class_names):
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		import seaborn as sns
		fig, ax = plt.subplots(figsize=(10, 8))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
		ax.set_xlabel('Predicted')
		ax.set_ylabel('True')
		ax.set_title('Confusion Matrix - LBP+SVM')
		out_dir = os.path.join(settings.BASE_DIR, 'attendance', 'models')
		os.makedirs(out_dir, exist_ok=True)
		fig.savefig(os.path.join(out_dir, 'lbp_svm_confusion_matrix.png'), bbox_inches='tight')
		plt.close(fig)
