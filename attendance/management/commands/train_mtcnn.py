from django.core.management.base import BaseCommand
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

class FaceDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        # Always resize to 224x224 and convert to tensor
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))
        image = Image.open(img_path).convert('RGB')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        w, h = image.size
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes) if boxes else np.zeros((0, 4))
        image = self.transform(image)
        return image, boxes, img_name

def detection_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    boxes = [item[1] for item in batch]  # list of arrays
    img_names = [item[2] for item in batch]
    return images, boxes, img_names

class Command(BaseCommand):
    help = 'Train MTCNN face detector using YOLO-format dataset.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting MTCNN training...'))

        # Parameters
        BATCH_SIZE = 32
        LEARNING_RATE = 1e-5
        EPOCHS = 20
        WEIGHT_DECAY = 0.05
        GRAD_CLIP = 1.0

        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stdout.write(self.style.SUCCESS(f'Using device: {device}'))

        # Paths
        train_images = 'datasets/face detection/images/train/'
        train_labels = 'datasets/face detection/labels/train/'
        val_images = 'datasets/face detection/images/val/'
        val_labels = 'datasets/face detection/labels/val/'

        # Datasets and loaders
        train_dataset = FaceDetectionDataset(train_images, train_labels)
        val_dataset = FaceDetectionDataset(val_images, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=detection_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=detection_collate_fn)

        # MTCNN setup (facenet-pytorch)
        mtcnn = MTCNN(keep_all=True, device=device)

        # Training logic will go here
        self.stdout.write(self.style.SUCCESS('Datasets and MTCNN initialized. Ready for training loop.'))

        best_f1 = 0
        for epoch in range(EPOCHS):
            mtcnn.train()
            running_loss = 0.0
            for images, boxes, _ in train_loader:
       
                pass 

            # Validation
            mtcnn.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for images, boxes, _ in val_loader:
                    for i, image in enumerate(images):
                        # Convert tensor to PIL image for MTCNN
                        if isinstance(image, torch.Tensor):
                            img_for_mtcnn = to_pil_image(image.cpu())
                        else:
                            img_for_mtcnn = image
                        try:
                            detected_boxes, _ = mtcnn.detect(img_for_mtcnn, landmarks=False)
                        except Exception as e:
                            detected_boxes = None
                        gt_boxes = boxes[i]
                        y_true.append(1 if len(gt_boxes) > 0 else 0)
                        y_pred.append(1 if detected_boxes is not None and len(detected_boxes) > 0 else 0)

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Face', 'Face'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix (Epoch {epoch+1})')
            matrix_path = f'training_matrix_epoch_{epoch+1}.png'
            plt.savefig(matrix_path)
            plt.close()
            self.stdout.write(self.style.SUCCESS(f'Saved confusion matrix: {matrix_path}'))

            # Calculate F1 for early stopping (optional)
            tp = cm[1, 1] if cm.shape == (2, 2) else 0
            fp = cm[0, 1] if cm.shape == (2, 2) else 0
            fn = cm[1, 0] if cm.shape == (2, 2) else 0
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                # Save the best model (simulated)
                torch.save(mtcnn.state_dict(), 'best_mtcnn.pth')
                self.stdout.write(self.style.SUCCESS('Saved best model.'))

        self.stdout.write(self.style.SUCCESS('Training complete.')) 