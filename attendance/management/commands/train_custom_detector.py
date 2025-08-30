from django.core.management.base import BaseCommand
from django.conf import settings
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pickle
from tqdm import tqdm
from typing import List, Tuple

# Torchvision detection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Dataset loader for YOLO-style detection
from attendance.utils.dataset_loader import (
	FaceDetectionYoloImageDataset,
	detection_collate_fn,
)


class Command(BaseCommand):
	help = "Train Faster R-CNN detector on YOLO-style face dataset (single class)."

	def add_arguments(self, parser):
		parser.add_argument('--epochs', type=int, default=30)
		parser.add_argument('--batch-size', type=int, default=4)
		parser.add_argument('--learning-rate', type=float, default=5e-4)
		parser.add_argument('--dataset-path', type=str, default='datasets/face detection')
		parser.add_argument('--val-split', type=float, default=0.2)
		parser.add_argument('--iou-threshold', type=float, default=0.5)

	def handle(self, *args, **options):
		epochs = options['epochs']
		batch_size = options['batch_size']
		lr = options['learning_rate']
		dataset_path = options['dataset_path']
		val_split = float(options['val_split'])
		iou_thr = float(options['iou_threshold'])

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

		full_ds = FaceDetectionYoloImageDataset(dataset_path)
		if len(full_ds) == 0:
			self.stdout.write(self.style.ERROR("No images found in YOLO detection dataset (check Images/ and labels/)."))
			return

		n_val = max(1, int(len(full_ds) * val_split))
		n_train = max(1, len(full_ds) - n_val)
		train_ds, val_ds = random_split(full_ds, [n_train, n_val])
		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=min(8, os.cpu_count() or 0), collate_fn=detection_collate_fn, pin_memory=torch.cuda.is_available())
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=min(8, os.cpu_count() or 0), collate_fn=detection_collate_fn, pin_memory=torch.cuda.is_available())

		model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
		num_classes = 2  # background + face
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
		model = model.to(device)

		# Reduce memory footprint without harming quality
		try:
			# Smaller processed image sizes to fit GPU memory while maintaining resolution
			if hasattr(model, 'transform'):
				model.transform.min_size = (512,)
				model.transform.max_size = 640
			# Reduce RPN proposals (train/test) to lower memory use
			if hasattr(model, 'rpn'):
				if hasattr(model.rpn, 'pre_nms_top_n_train'):
					model.rpn.pre_nms_top_n_train = 1000
				if hasattr(model.rpn, 'pre_nms_top_n_test'):
					model.rpn.pre_nms_top_n_test = 600
				if hasattr(model.rpn, 'post_nms_top_n_train'):
					model.rpn.post_nms_top_n_train = 1000
				if hasattr(model.rpn, 'post_nms_top_n_test'):
					model.rpn.post_nms_top_n_test = 600
		except Exception:
			pass

		optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=1e-4)

		train_losses: List[float] = []
		val_losses: List[float] = []
		val_precisions: List[float] = []
		val_recalls: List[float] = []
		val_f1s: List[float] = []
		val_accuracies: List[float] = []
		train_precisions: List[float] = []
		train_recalls: List[float] = []
		train_f1s: List[float] = []
		train_accuracies: List[float] = []
		best_f1: float = 0.0

		scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
		for epoch in range(epochs):
			model.train()
			total_loss = 0.0
			for images, targets in tqdm(train_loader, total=len(train_loader), desc=f"Train {epoch+1}/{epochs}", dynamic_ncols=True, leave=False):
				images = [img.to(device, non_blocking=True) for img in images]
				targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
				with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
					loss_dict = model(images, targets)
					loss = sum(loss for loss in loss_dict.values())
				optimizer.zero_grad(set_to_none=True)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				total_loss += float(loss.item())
			train_losses.append(total_loss / max(1, len(train_loader)))

			# Validation: compute simple detection precision/recall/F1@IoU
			model.eval()
			val_total_loss = 0.0
			tp = 0
			fp = 0
			fn = 0
			with torch.no_grad():
				for images, targets in tqdm(val_loader, total=len(val_loader), desc=f"Val {epoch+1}/{epochs}", dynamic_ncols=True, leave=False):
					images = [img.to(device, non_blocking=True) for img in images]
					targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
					# Compute a validation loss by enabling targets (evaluation loss proxy)
					with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
						loss_dict = model(images, targets_dev)
						val_total_loss += float(sum(loss for loss in loss_dict.values()).item())
					# Predictions
					preds = model(images)
					for pred, tgt in zip(preds, targets):
						pboxes = pred.get('boxes', torch.empty((0, 4), device=device)).detach().cpu()
						pscores = pred.get('scores', torch.empty((0,), device=device)).detach().cpu()
						keep = pscores >= 0.5
						pboxes = pboxes[keep]
						gboxes = tgt.get('boxes', torch.empty((0, 4))).detach().cpu()
						m_tp, m_fp, m_fn = self._match_counts(pboxes, gboxes, iou_thr)
						tp += m_tp
						fp += m_fp
						fn += m_fn

			val_losses.append(val_total_loss / max(1, len(val_loader)))
			prec = float(tp) / float(max(1, tp + fp))
			rec = float(tp) / float(max(1, tp + fn))
			f1 = 2.0 * prec * rec / max(1e-8, (prec + rec))
			acc = float(tp) / float(max(1, tp + fp + fn))
			val_precisions.append(prec)
			val_recalls.append(rec)
			val_f1s.append(f1)
			val_accuracies.append(acc)
			self.stdout.write(f"Epoch {epoch+1}/{epochs} - loss {train_losses[-1]:.4f} - val_loss {val_losses[-1]:.4f} - P {prec:.3f} R {rec:.3f} F1 {f1:.3f}")

			if f1 > best_f1:
				best_f1 = f1
				self._save_model(model, 'best_detection_model.pth')

		# Save final model and history
		self._save_model(model, 'final_detection_model.pth')
		history = {
			'train_losses': train_losses,
			'val_losses': val_losses,
			'train_precisions': train_precisions,
			'train_recalls': train_recalls,
			'train_f1s': train_f1s,
			'train_accuracies': train_accuracies,
			'val_precisions': val_precisions,
			'val_recalls': val_recalls,
			'val_f1s': val_f1s,
			'val_accuracies': val_accuracies,
			'class_names': ['background', 'face'],
			'best_val_f1': best_f1,
		}
		self._save_history(history, 'detection_training_history.pkl')
		self.stdout.write(self.style.SUCCESS(f"Done. Best val F1: {best_f1:.3f}"))

	def _save_model(self, model, filename):
		models_dir = os.path.join(settings.BASE_DIR, 'attendance', 'models')
		os.makedirs(models_dir, exist_ok=True)
		path = os.path.join(models_dir, filename)
		torch.save(model.state_dict(), path)
		self.stdout.write(f"Saved model -> {path}")

	def _save_history(self, history, filename):
		models_dir = os.path.join(settings.BASE_DIR, 'attendance', 'models')
		os.makedirs(models_dir, exist_ok=True)
		path = os.path.join(models_dir, filename)
		with open(path, 'wb') as f:
			pickle.dump(history, f)
		self.stdout.write(f"Saved history -> {path}")

	@staticmethod
	def _match_counts(pboxes: torch.Tensor, gboxes: torch.Tensor, iou_thr: float) -> Tuple[int, int, int]:
		"""Greedy IoU matching to compute TP, FP, FN per image."""
		if pboxes.numel() == 0 and gboxes.numel() == 0:
			return 0, 0, 0
		if pboxes.numel() == 0:
			return 0, 0, int(gboxes.shape[0])
		if gboxes.numel() == 0:
			return 0, int(pboxes.shape[0]), 0
		iou = Command._box_iou(pboxes, gboxes)
		matched_g = set()
		matched_p = set()
		# Greedy match
		while True:
			max_val, max_idx = (iou if iou.numel() > 0 else torch.zeros((1,))).max(dim=None)
			if iou.numel() == 0 or float(max_val) < iou_thr:
				break
			pi, gi = torch.div(max_idx, iou.shape[1], rounding_mode='floor'), max_idx % iou.shape[1]
			matched_p.add(int(pi))
			matched_g.add(int(gi))
			iou[pi, :] = -1.0
			iou[:, gi] = -1.0
		tp = len(matched_p)
		fp = int(pboxes.shape[0]) - tp
		fn = int(gboxes.shape[0]) - len(matched_g)
		return tp, fp, fn

	@staticmethod
	def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
		# boxes in [x1,y1,x2,y2]
		area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
		area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
		lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
		rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
		wh = (rb - lt).clamp(min=0)
		inter = wh[:, :, 0] * wh[:, :, 1]
		union = area1[:, None] + area2 - inter
		return inter / (union + 1e-6)
