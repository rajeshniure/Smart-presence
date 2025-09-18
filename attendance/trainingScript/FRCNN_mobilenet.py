import os
import glob
import torch
import torchvision
from torchvision import tv_tensors
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

HYPERPARAMS = {
    'learning_rate': 1e-4, 
    'weight_decay': 1e-4, 
    'num_epochs': 50,
    'batch_size': 8, 
    'accumulation_steps': 2, 
    'train_split_ratio': 0.8,
    'num_workers': 2, 
    'max_grad_norm': 1.0, 
    'warmup_epochs': 1
}

class FaceDataset(Dataset):
    def __init__(self, image_paths, label_dir, transforms=None):
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        label_file = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_c, y_c, w, h = map(float, parts)
                        if int(cls) == 0:
                            xmin = (x_c - w / 2) * width
                            ymin = (y_c - h / 2) * height
                            xmax = (x_c + w / 2) * width
                            ymax = (y_c + h / 2) * height
                            if xmin < xmax and ymin < ymax:
                                boxes.append([xmin, ymin, xmax, ymax])
                                labels.append(1)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes_tensor,
            format="XYXY",
            canvas_size=(height, width)
        )
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        if boxes:
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            target["area"] = area
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

def get_transform(train):
    transforms = [T.ToImage()]
    if train:
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        ])
    transforms.extend([
        T.ToDtype(torch.float32, scale=True),
        T.Resize((224, 224), antialias=True),
        T.SanitizeBoundingBoxes()
    ])
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

# --- Training & Validation Loops ---
def train_one_epoch(model, optimizer, data_loader, device, epoch, accumulation_steps, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Training")
    optimizer.zero_grad()
    for i, (images, targets) in enumerate(progress_bar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        if not torch.isfinite(losses):
            print(f"Warning: Non-finite loss detected. Skipping update.")
            continue
        scaler.scale(losses / accumulation_steps).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=HYPERPARAMS['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += losses.item()
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")
    if len(data_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=HYPERPARAMS['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

@torch.inference_mode()
def validate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy')
    metric.to(device)
    for images, targets in tqdm(data_loader, desc="Validation"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = model(images)
        metric.update(predictions, targets)
    return metric.compute()

def plot_metrics(history, output_dir):
    epochs = range(1, len(history['train_losses']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_losses'], 'b-o', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    val_map = [m['map'].item() for m in history['val_metrics']]
    val_map50 = [m['map_50'].item() for m in history['val_metrics']]
    plt.plot(epochs, val_map, 'r-o', label='mAP @ .50-.95')
    plt.plot(epochs, val_map50, 'c-^', label='mAP @ .50')
    plt.title('Validation mAP')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_plots.png'))
    plt.close()

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dir = '/kaggle/input/labelimages/datasets/Images'
    label_dir = '/kaggle/input/labelimages/datasets/labels'
    output_dir = '/kaggle/working/'
    os.makedirs(output_dir, exist_ok=True)

    all_image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    train_size = int(HYPERPARAMS['train_split_ratio'] * len(all_image_paths))
    train_paths = all_image_paths[:train_size]
    val_paths = all_image_paths[train_size:]

    train_dataset = FaceDataset(train_paths, label_dir, transforms=get_transform(train=True))
    val_dataset = FaceDataset(val_paths, label_dir, transforms=get_transform(train=False))

    print(f"Found {len(all_image_paths)} total images.")
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=True, num_workers=HYPERPARAMS['num_workers'], collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=False, num_workers=HYPERPARAMS['num_workers'], collate_fn=collate_fn, pin_memory=True)

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HYPERPARAMS['learning_rate'], weight_decay=HYPERPARAMS['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HYPERPARAMS['num_epochs'] - HYPERPARAMS['warmup_epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history = {'train_losses': [], 'val_metrics': []}
    print(f"Starting training on {device}...")
    for epoch in range(HYPERPARAMS['num_epochs']):
        if epoch < HYPERPARAMS['warmup_epochs'] and HYPERPARAMS['warmup_epochs'] > 0:
            warmup_factor = (epoch + 1) / HYPERPARAMS['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = HYPERPARAMS['learning_rate'] * warmup_factor
        
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, HYPERPARAMS['accumulation_steps'], scaler)
        history['train_losses'].append(train_loss)
        val_metrics = validate(model, val_loader, device)
        history['val_metrics'].append(val_metrics)
        
        if epoch >= HYPERPARAMS['warmup_epochs']:
            lr_scheduler.step()
        
        print(f"Epoch [{epoch + 1}/{HYPERPARAMS['num_epochs']}] | Train Loss: {train_loss:.4f} | Val mAP: {val_metrics['map']:.4f} | Val mAP@50: {val_metrics['map_50']:.4f}")

    print("\n Training complete. Saving results...")
    final_metrics = history['val_metrics'][-1]
    print("\nFinal Validation Metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, torch.Tensor): print(f"  {k}: {v.item():.4f}")
    
    serializable_history = {'train_losses': history['train_losses'], 'val_metrics': [{k: v.cpu().tolist() for k, v in m.items()} for m in history['val_metrics']]}
    with open(os.path.join(output_dir, 'metrics_history.json'), 'w') as f:
        json.dump(serializable_history, f, indent=4)
        
    plot_metrics(history, output_dir)
    model_save_path = os.path.join(output_dir, 'mobilenet_1_0_224_tf.h5')
    torch.save(model.state_dict(), model_save_path)
    print(f"\n Model and results saved in: {output_dir}")