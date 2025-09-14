import os
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# =========================
# Custom ResNet Architecture
# =========================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# =========================
# Config
# =========================
DATASET_DIR = "/kaggle/input/faceimages/face_recognition"
OUTPUT_DIR = "/kaggle/working"

EPOCHS = 30
BATCH_SIZE = 128  # Increased for multi-GPU
LEARNING_RATE = 2e-2  # Scaled for multi-GPU (2 GPUs * 1e-2)
VAL_INTERVAL = 1
PATIENCE = 6
SEED = 42
NUM_CLASSES_TO_PLOT = 10  # Number of classes to include in confusion matrix

# =========================
# Setup
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPU(s)")

# =========================
# Data
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
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

full_dataset = datasets.ImageFolder(DATASET_DIR)
if len(full_dataset) == 0:
    raise RuntimeError(f"No images found under {DATASET_DIR}. Expected subfolders per class.")

# Stratified split
targets = np.array(full_dataset.targets)
indices = list(range(len(full_dataset)))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=targets
)

train_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_tf)
val_dataset = datasets.ImageFolder(DATASET_DIR, transform=val_tf)

train_ds = torch.utils.data.Subset(train_dataset, train_indices)
val_ds = torch.utils.data.Subset(val_dataset, val_indices)

num_workers = min(8, os.cpu_count() or 0)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers > 0)
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers > 0)
)

class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# Select top N classes for confusion matrix based on sample count
labels = np.array([train_dataset.targets[idx] for idx in train_indices])
counts = np.bincount(labels, minlength=num_classes)
top_class_indices = np.argsort(counts)[-NUM_CLASSES_TO_PLOT:][::-1]
top_class_names = [class_names[i] for i in top_class_indices]
print(f"Top {NUM_CLASSES_TO_PLOT} classes for confusion matrix: {top_class_names}")

# =========================
# Model (Custom ResNet18 from scratch)
# =========================
model = resnet18(num_classes=num_classes)
model = model.to(device)
if num_gpus > 1:
    model = nn.DataParallel(model)
    print("Model wrapped with DataParallel for multi-GPU training")

# =========================
# Loss / Optimizer
# =========================
counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
weights_arr = 1.0 / (counts + 1e-6)
weights_arr = weights_arr / weights_arr.sum() * num_classes
class_weights = torch.tensor(weights_arr, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# Helpers
# =========================
def save_model(model, class_names, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    # Save the state_dict of the unwrapped model if using DataParallel
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({'state_dict': state_dict, 'class_names': class_names, 'arch': 'resnet18'}, path)
    print(f"Saved model -> {path}")

def save_history(history, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved history -> {path}")

def plot_line(x_train, train_data, train_label, x_val=None, val_data=None, val_label=None, title='', ylabel='', filename=''):
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, train_data, label=train_label)
    if x_val is not None and val_data is not None:
        plt.plot(x_val, val_data, label=val_label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved plot -> {os.path.join(OUTPUT_DIR, filename)}")

def plot_val_metrics(x_val, precisions, recalls, f1s, title='Validation Metrics', filename='val_metrics.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(x_val, precisions, label='Precision')
    plt.plot(x_val, recalls, label='Recall')
    plt.plot(x_val, f1s, label='F1 Score')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved plot -> {os.path.join(OUTPUT_DIR, filename)}")

def plot_confusion_matrix(cm, class_names, class_indices, title='Confusion Matrix', cmap=plt.cm.Blues, filename='confusion_matrix.png'):
    cm_subset = cm[np.ix_(class_indices, class_indices)]
    fig, ax = plt.subplots(figsize=(max(10, len(class_indices)), max(10, len(class_indices))))
    im = ax.imshow(cm_subset, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm_subset.shape[1]),
           yticks=np.arange(cm_subset.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm_subset.max() / 2.
    for i in range(cm_subset.shape[0]):
        for j in range(cm_subset.shape[1]):
            ax.text(j, i, format(cm_subset[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_subset[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved plot -> {os.path.join(OUTPUT_DIR, filename)}")

# =========================
# Training Loop
# =========================
best_acc = 0.0
no_improve = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
val_precisions, val_recalls, val_f1s = [], [], []
val_cms, val_preds_history, val_targets_history = [], [], []

scaler_enabled = torch.cuda.is_available()

for epoch in range(EPOCHS):
    model.train()
    total_loss, train_correct, train_total = 0.0, 0, 0
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    for data, target in tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda' if scaler_enabled else 'cpu', enabled=scaler_enabled):
            logits = model(data)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        train_total += target.size(0)
        train_correct += (pred == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)

    scheduler.step()

    # =========================
    # Validation
    # =========================
    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()
        correct, total, val_total_loss = 0, 0, 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Val {epoch+1}/{EPOCHS}", leave=False):
                data, target = data.to(device), target.to(device)
                with torch.amp.autocast(device_type='cuda' if scaler_enabled else 'cpu', enabled=scaler_enabled):
                    logits = model(data)
                    vloss = criterion(logits, target)

                val_total_loss += vloss.item()
                pred = logits.argmax(dim=1)
                total += target.size(0)
                correct += (pred == target).sum().item()
                all_preds.extend(pred.cpu().tolist())
                all_targets.extend(target.cpu().tolist())

        acc = 100.0 * correct / total if total > 0 else 0.0
        val_loss_avg = val_total_loss / len(val_loader)
        val_losses.append(val_loss_avg)
        val_accuracies.append(acc)

        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
        val_precisions.append(prec)
        val_recalls.append(rec)
        val_f1s.append(f1)

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
        val_cms.append(cm)
        val_preds_history.append(all_preds)
        val_targets_history.append(all_targets)

        print(f"Epoch {epoch+1}/{EPOCHS} - loss {avg_loss:.4f} - train_acc {train_acc:.2f}% - "
              f"val_loss {val_loss_avg:.4f} - val_acc {acc:.2f}% - f1 {f1:.3f}")

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            save_model(model, class_names, 'best_recognition_model.pth')
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

# =========================
# Save final artifacts
# =========================
save_model(model, class_names, 'final_recognition_model.pth')
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'val_precisions': val_precisions,
    'val_recalls': val_recalls,
    'val_f1s': val_f1s,
    'val_confusion_matrices': val_cms,
    'val_preds': val_preds_history,
    'val_targets': val_targets_history,
    'best_val_acc': best_acc,
    'class_names': class_names,
}
save_history(history, 'recognition_training_history.pkl')
print(f"Done. Best val acc: {best_acc:.2f}%")

# =========================
# Visualization and Plots
# =========================
x_train = np.arange(1, len(history['train_losses']) + 1)
x_val = np.arange(VAL_INTERVAL, len(history['train_losses']) + 1, VAL_INTERVAL)

# Loss plot
plot_line(x_train, history['train_losses'], 'Train Loss', x_val, history['val_losses'], 'Val Loss',
          title='Training and Validation Loss', ylabel='Loss', filename='loss_plot.png')

# Accuracy plot
plot_line(x_train, history['train_accuracies'], 'Train Accuracy', x_val, history['val_accuracies'], 'Val Accuracy',
          title='Training and Validation Accuracy', ylabel='Accuracy (%)', filename='accuracy_plot.png')

# Validation precision, recall, f1 plot
plot_val_metrics(x_val, history['val_precisions'], history['val_recalls'], history['val_f1s'])

# Confusion matrix for the best validation accuracy, using top classes
best_index = np.argmax(history['val_accuracies'])
best_cm = history['val_confusion_matrices'][best_index]
plot_confusion_matrix(best_cm, top_class_names, top_class_indices,
                      title=f'Best Validation Confusion Matrix (Top {NUM_CLASSES_TO_PLOT} Classes)')

print("All plots generated.")