from django.core.management.base import BaseCommand
from django.conf import settings
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Use torchvision ImageFolder with strong, face-appropriate augmentations and transfer learning
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


class Command(BaseCommand):
	help = "Train face recognition with transfer learning (ResNet18) on class folders."

	def add_arguments(self, parser):
		parser.add_argument('--epochs', type=int, default=50)
		parser.add_argument('--batch-size', type=int, default=32)
		parser.add_argument('--learning-rate', type=float, default=1e-3)
		parser.add_argument('--dataset-path', type=str, default='datasets/face recognition/face_recognition_datasets')

	def handle(self, *args, **options):
		epochs = int(options['epochs'])
		batch_size = int(options['batch_size'])
		lr = float(options['learning_rate'])
		dataset_path = options['dataset_path']

		if not os.path.isdir(dataset_path):
			self.stdout.write(self.style.ERROR(f"Dataset path not found: {dataset_path}"))
			return

		# Enable cuDNN autotuner for performance
		if torch.cuda.is_available():
			try:
				import torch.backends.cudnn as cudnn
				cudnn.benchmark = True
			except Exception:
				pass
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Transforms tuned for faces: modest augmentation, ImageNet normalization
		# ImageNet normalization constants to avoid version-dependent meta access
		IMAGENET_MEAN = [0.485, 0.456, 0.406]
		IMAGENET_STD = [0.229, 0.224, 0.225]

		train_tf = transforms.Compose([
			transforms.Resize(256),
			transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		])
		val_tf = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		])

		full_train = datasets.ImageFolder(dataset_path, transform=train_tf)
		# Split for val to mimic previous behavior (80/20)
		n_val = max(1, int(0.2 * len(full_train)))
		n_train = max(1, len(full_train) - n_val)
		train_ds, val_ds = torch.utils.data.random_split(full_train, [n_train, n_val])
		# Override val transform
		val_ds.dataset = datasets.ImageFolder(dataset_path, transform=val_tf)

		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=min(8, os.cpu_count() or 0))
		val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=min(8, os.cpu_count() or 0))

		class_names = full_train.classes
		num_classes = len(class_names)

		# Build model
		weights = ResNet18_Weights.IMAGENET1K_V1
		model = resnet18(weights=weights)
		in_features = model.fc.in_features
		model.fc = nn.Linear(in_features, num_classes)
		model = model.to(device)

		# Compute class weights from train split for imbalance
		labels = [lbl for _, lbl in [full_train.samples[i] for i in train_ds.indices]]
		counts = np.bincount(np.array(labels), minlength=num_classes)
		weights_arr = 1.0 / (counts + 1e-6)
		weights_arr = weights_arr / weights_arr.sum() * num_classes
		class_weights = torch.tensor(weights_arr, dtype=torch.float32).to(device)
		criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
		optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

		best_acc = 0.0
		train_losses = []
		val_losses = []
		train_accuracies = []
		val_accuracies = []
		train_f1s = []
		val_f1s = []
		val_precisions = []
		val_recalls = []
		val_cms = []
		train_cms = []

		for epoch in range(epochs):
			model.train()
			total_loss = 0.0
			train_correct = 0
			train_total = 0
			scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
			all_train_preds = []
			all_train_targets = []
			for data, target in tqdm(train_loader, total=len(train_loader), desc=f"Train {epoch+1}/{epochs}", dynamic_ncols=True, leave=False):
				data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)
				with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
					logits = model(data)
					loss = criterion(logits, target)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				total_loss += float(loss.item())
				pred = logits.argmax(dim=1)
				train_total += int(target.size(0))
				train_correct += int((pred == target).sum().item())
				all_train_preds.extend(pred.detach().cpu().tolist())
				all_train_targets.extend(target.detach().cpu().tolist())

			avg_loss = total_loss / max(1, len(train_loader))
			train_losses.append(avg_loss)
			train_acc = 100.0 * train_correct / max(1, train_total)
			train_accuracies.append(train_acc)
			# Binary training confusion (recognized vs not)
			binary_true_train = [1] * len(all_train_targets)
			binary_pred_train = [1 if p == t else 0 for p, t in zip(all_train_preds, all_train_targets)]
			cm_train = confusion_matrix(binary_true_train, binary_pred_train, labels=[0, 1])
			train_cms.append(cm_train)

			# Validation
			model.eval()
			correct = 0
			total = 0
			val_total_loss = 0.0
			all_preds = []
			all_targets = []
			with torch.no_grad():
				for data, target in tqdm(val_loader, total=len(val_loader), desc=f"Val {epoch+1}/{epochs}", dynamic_ncols=True, leave=False):
					data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
					with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
						logits = model(data)
						vloss = criterion(logits, target)
					val_total_loss += float(vloss.item())
					pred = logits.argmax(dim=1)
					total += int(target.size(0))
					correct += int((pred == target).sum().item())
					all_preds.extend(pred.detach().cpu().tolist())
					all_targets.extend(target.detach().cpu().tolist())

			acc = 100.0 * correct / max(1, total)
			val_accuracies.append(acc)
			val_loss_avg = val_total_loss / max(1, len(val_loader))
			val_losses.append(val_loss_avg)
			prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
			val_precisions.append(float(prec))
			val_recalls.append(float(rec))
			val_f1s.append(float(f1))
			# Binary confusion matrix for visualization
			binary_true = [1] * len(all_targets)
			binary_pred = [1 if p == t else 0 for p, t in zip(all_preds, all_targets)]
			cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
			val_cms.append(cm)
			scheduler.step(acc)
			self.stdout.write(f"Epoch {epoch+1}/{epochs} - loss {avg_loss:.4f} - train_acc {train_acc:.2f}% - val_loss {val_loss_avg:.4f} - val_acc {acc:.2f}% - f1 {f1:.3f}")

			if acc > best_acc:
				best_acc = acc
				self._save_model(model, class_names, 'best_recognition_model.pth')

		# Save final model and history
		self._save_model(model, class_names, 'final_recognition_model.pth')
		history = {
			'train_losses': train_losses,
			'val_losses': val_losses,
			'train_accuracies': train_accuracies,
			'val_accuracies': val_accuracies,
			'train_f1s': train_f1s,
			'val_f1s': val_f1s,
			'val_precisions': val_precisions,
			'val_recalls': val_recalls,
			'val_confusion_matrices': val_cms,
			'train_confusion_matrices': train_cms,
			'best_val_acc': best_acc,
			'class_names': ['not_recognized','recognized'],
		}
		self._save_history(history, 'recognition_training_history.pkl')
		self.stdout.write(self.style.SUCCESS(f"Done. Best val acc: {best_acc:.2f}%"))

	def _save_model(self, model, class_names, filename):
		models_dir = os.path.join(settings.BASE_DIR, 'attendance', 'models')
		os.makedirs(models_dir, exist_ok=True)
		path = os.path.join(models_dir, filename)
		torch.save({ 'state_dict': model.state_dict(), 'class_names': class_names, 'arch': 'resnet18' }, path)
		self.stdout.write(f"Saved model -> {path}")

	def _save_history(self, history, filename):
		models_dir = os.path.join(settings.BASE_DIR, 'attendance', 'models')
		os.makedirs(models_dir, exist_ok=True)
		path = os.path.join(models_dir, filename)
		with open(path, 'wb') as f:
			pickle.dump(history, f)
		self.stdout.write(f"Saved history -> {path}")
