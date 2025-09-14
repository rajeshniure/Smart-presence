import os
import glob
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset, random_split
# Use the new 'v2' API for more features and consistency
import torchvision.transforms.v2 as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
# Use torchmetrics for standardized and correct evaluation
from torchmetrics.detection import MeanAveragePrecision

# Tuned for better stability and performance
HYPERPARAMS = {
    'learning_rate': 1e-4,          
    'weight_decay': 1e-4,
    'lr_scheduler_step_size': 8,    
    'lr_scheduler_gamma': 0.1,      
    'num_epochs': 50,
    'batch_size': 4,                
    'accumulation_steps': 4,       
    'train_split_ratio': 0.8,
    'num_workers': 2,              
    'max_grad_norm': 1.0           
}

# --- Dataset Definition ---
class FaceDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        label_file = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_center, y_center, w, h = map(float, parts)
                        # Assuming class 0 is 'face' in YOLO format
                        if cls == 0:
                            xmin = (x_center - w / 2) * width
                            ymin = (y_center - h / 2) * height
                            xmax = (x_center + w / 2) * width
                            ymax = (y_center + h / 2) * height
                            # Ensure the box has a valid area
                            if xmin < xmax and ymin < ymax:
                                boxes.append([xmin, ymin, xmax, ymax])
                                labels.append(1) # Class 1 for 'face' (0 is background)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.empty((0,)),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }

        if self.transforms:
            # The v2 transform API handles images and targets together
            img, target = self.transforms(img, target)

        return img, target

# --- Transformations ---
def get_transform(train):
    """
    Defines the image transformations. Includes augmentation for the training set.
    """
    transforms = []
    if train:
        # For training, add data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    # **FIX**: Convert the PIL image to a PyTorch tensor BEFORE other tensor operations
    transforms.append(T.ToImage())
    
    # Converts tensor dtype to float and scales values to the [0, 1] range
    transforms.append(T.ToDtype(torch.float32, scale=True))
    
    # Normalize the tensor with standard ImageNet stats
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    """
    return tuple(zip(*batch))

# --- Training & Validation Loops ---
def train_one_epoch(model, optimizer, data_loader, device, epoch, accumulation_steps):
    """
    Efficiently trains the model for one epoch with gradient accumulation.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Training")
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(progress_bar):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses):
            print(f"Warning: NaN loss detected at batch {i}. Skipping update.")
            continue

        # Scale loss for gradient accumulation
        (losses / accumulation_steps).backward()

        # Perform optimizer step after accumulating gradients for a few steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=HYPERPARAMS['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())
        
    return total_loss / len(data_loader)

@torch.inference_mode()
def validate(model, data_loader, device):
    """
    Validates the model on the validation set using torchmetrics for standard mAP calculation.
    """
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.to(device)
    
    progress_bar = tqdm(data_loader, desc="Validation")
    for images, targets in progress_bar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)
        
        # Update metric with predictions and ground truth
        metric.update(predictions, targets)

    # Compute final metrics over the entire validation set
    results = metric.compute()
    return results

# --- Visualization ---
def plot_metrics(history, output_dir):
    """
    Plots training loss and validation metrics and saves them to files.
    """
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_losses'], 'b-o', label='Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot Validation mAP (Mean Average Precision) and mAR (Mean Average Recall)
    plt.figure(figsize=(10, 5))
    val_map = [m['map'].item() for m in history['val_metrics']]
    val_mar = [m['mar_100'].item() for m in history['val_metrics']]
    plt.plot(epochs, val_map, 'r-o', label='Validation mAP @ IoU .50-.95')
    plt.plot(epochs, val_mar, 'g-s', label='Validation mAR @ 100 Detections')
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'validation_metrics_plot.png'))
    plt.close()

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define paths - **Please adjust these to your environment**
    image_dir = '/kaggle/input/labelimages/datasets/Images'
    label_dir = '/kaggle/input/labelimages/datasets/labels'
    output_dir = '/kaggle/working/'
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading ---
    # Create two versions of the dataset: one with augmentations (for training)
    # and one without (for validation)
    dataset_train_aug = FaceDataset(image_dir, label_dir, transforms=get_transform(train=True))
    dataset_val_no_aug = FaceDataset(image_dir, label_dir, transforms=get_transform(train=False))

    train_size = int(HYPERPARAMS['train_split_ratio'] * len(dataset_train_aug))
    val_size = len(dataset_train_aug) - train_size
    
    # Use indices to ensure the same images are split, but sourced from different
    # dataset objects (one with augmentation, one without)
    indices = torch.randperm(len(dataset_train_aug)).tolist()
    train_dataset = torch.utils.data.Subset(dataset_train_aug, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset_val_no_aug, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=True, num_workers=HYPERPARAMS['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=False, num_workers=HYPERPARAMS['num_workers'], collate_fn=collate_fn)

    # --- Model Setup ---
    # Load a pre-trained model
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one for our number of classes (2 = background + face)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # --- Optimizer & Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=HYPERPARAMS['learning_rate'], weight_decay=HYPERPARAMS['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=HYPERPARAMS['lr_scheduler_step_size'], gamma=HYPERPARAMS['lr_scheduler_gamma'])

    # --- Training Loop ---
    history = {'train_losses': [], 'val_metrics': []}

    print(f"🚀 Starting training on {device}...")
    for epoch in range(HYPERPARAMS['num_epochs']):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, HYPERPARAMS['accumulation_steps'])
        history['train_losses'].append(train_loss)
        
        val_metrics = validate(model, val_loader, device)
        history['val_metrics'].append(val_metrics)
        
        lr_scheduler.step()

        print(f"Epoch [{epoch + 1}/{HYPERPARAMS['num_epochs']}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val mAP: {val_metrics['map']:.4f} | "
              f"Val mAR: {val_metrics['mar_100']:.4f}")

    # --- Save Results ---
    # Convert tensors in history to lists for clean JSON serialization
    serializable_history = {
        'train_losses': history['train_losses'],
        'val_metrics': [{k: v.tolist() for k, v in m.items()} for m in history['val_metrics']]
    }
    with open(os.path.join(output_dir, 'metrics_history.json'), 'w') as f:
        json.dump(serializable_history, f, indent=4)

    plot_metrics(history, output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, 'faster_rcnn_face_model_optimized.pth'))

    print("\n Training complete.")
    print(f" Model saved to: {os.path.join(output_dir, 'faster_rcnn_face_model_optimized.pth')}")
    print(f" Metrics and plots saved in: {output_dir}")