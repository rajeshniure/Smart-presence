import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    # Prefer torchvision transforms if available (installed via requirements)
    from torchvision.transforms import functional as F
except Exception:  # pragma: no cover
    F = None  # type: ignore


class FaceDetectionDataset(Dataset):
    """Detection dataset supporting two structures under root:
    1) YOLO-style labels: root/Images/*.jpg and root/labels/<same_name>.txt
       Produces positive crops from labeled faces and random negatives.
    2) Binary folders: root/Images/faces/*.jpg and root/Images/non_faces/*.jpg
    """

    def __init__(self, root_dir: str, negatives_per_image: int = 2) -> None:
        self.samples: List[Tuple[str, Tuple[int, int, int, int], int]] = []  # (img_path, bbox, label)
        images_dir = os.path.join(root_dir, 'Images')
        labels_dir = os.path.join(root_dir, 'labels')

        if os.path.isdir(labels_dir):
            self._load_yolo_format(images_dir, labels_dir, negatives_per_image)
        else:
            self._load_binary_folders(images_dir)

    def _load_binary_folders(self, images_dir: str) -> None:
        faces_dir = os.path.join(images_dir, 'faces')
        non_faces_dir = os.path.join(images_dir, 'non_faces')
        if os.path.isdir(faces_dir):
            for name in os.listdir(faces_dir):
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(faces_dir, name), (0, 0, -1, -1), 1))
        if os.path.isdir(non_faces_dir):
            for name in os.listdir(non_faces_dir):
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(non_faces_dir, name), (0, 0, -1, -1), 0))

    def _load_yolo_format(self, images_dir: str, labels_dir: str, negatives_per_image: int) -> None:
        import random
        img_exts = ('.jpg', '.jpeg', '.png')
        for name in os.listdir(images_dir):
            if not name.lower().endswith(img_exts):
                continue
            img_path = os.path.join(images_dir, name)
            base = os.path.splitext(name)[0]
            label_path = os.path.join(labels_dir, base + '.txt')
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue

            boxes = []
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            cls, x_c, y_c, bw, bh = parts
                            cls_id = int(float(cls))
                            # Convert normalized YOLO to pixel box [x1,y1,w,h]
                            x_center = float(x_c) * w
                            y_center = float(y_c) * h
                            box_w = float(bw) * w
                            box_h = float(bh) * h
                            x1 = max(0, int(x_center - box_w / 2))
                            y1 = max(0, int(y_center - box_h / 2))
                            box = (x1, y1, int(box_w), int(box_h))
                            # Only one class assumed: face
                            if cls_id == 0:
                                boxes.append(box)
                                self.samples.append((img_path, box, 1))
                except Exception:
                    pass

            # Generate negatives by sampling windows with low overlap
            if negatives_per_image > 0:
                import numpy as np
                for _ in range(negatives_per_image):
                    if w < 64 or h < 64:
                        break
                    win_w = random.randint(48, min(128, w))
                    win_h = random.randint(48, min(128, h))
                    x1 = random.randint(0, max(0, w - win_w))
                    y1 = random.randint(0, max(0, h - win_h))
                    candidate = (x1, y1, win_w, win_h)
                    if self._iou_low(candidate, boxes):
                        self.samples.append((img_path, candidate, 0))

    def _iou_low(self, cand: Tuple[int, int, int, int], boxes: List[Tuple[int, int, int, int]], thr: float = 0.1) -> bool:
        if not boxes:
            return True
        def iou(a, b):
            ax1, ay1, aw, ah = a
            bx1, by1, bw, bh = b
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh
            inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0, min(ay2, by2) - max(ay1, by1))
            inter = inter_w * inter_h
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0
        return all(iou(cand, b) < thr for b in boxes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, box, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if box[2] > 0 and box[3] > 0:
            x, y, w, h = box
            img = img.crop((x, y, x + w, y + h))
        img = img.resize((64, 64))
        arr = np.array(img)
        tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
        return tensor, int(label)


class FaceRecognitionDataset(Dataset):
    """Recognition dataset expecting class subfolders:
    root/<person_name>/*.jpg
    """

    def __init__(self, dataset_dir: str) -> None:
        self.class_names: List[str] = []
        self.samples: List[Tuple[str, int]] = []
        for class_name in sorted(os.listdir(dataset_dir)):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_names.append(class_name)
            class_idx = len(self.class_names) - 1
            for name in os.listdir(class_dir):
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, name), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        # Lightweight augmentations
        try:
            import random
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Random crop (90-100%) and resize
            w, h = img.size
            scale = 0.9 + random.random() * 0.1
            nw, nh = int(w * scale), int(h * scale)
            if nw < w and nh < h:
                x0 = random.randint(0, w - nw)
                y0 = random.randint(0, h - nh)
                img = img.crop((x0, y0, x0 + nw, y0 + nh))
        except Exception:
            pass
        # Use a slightly larger input for better capacity
        img = img.resize((96, 96))
        arr = np.array(img)
        tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
        # Normalize
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, int(label)

    def get_class_names(self) -> List[str]:
        return self.class_names


def split_loaders(dataset: Dataset, batch_size: int = 32, train_split: float = 0.8):
    n_train = int(train_split * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    use_cuda = torch.cuda.is_available()
    pin = True if use_cuda else False
    num_workers = min(8, os.cpu_count() or 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=num_workers)
    return train_loader, val_loader


class FaceDetectionYoloImageDataset(Dataset):
    """Object-detection dataset that returns full image + targets from YOLO labels.

    Expects structure:
      root/Images/*.jpg (or .png)
      root/labels/<same_name>.txt  (YOLO: class x_center y_center width height)

    Single class (face) assumed: class id 0.
    """

    def __init__(self, root_dir: str) -> None:
        self.images_dir = os.path.join(root_dir, 'Images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.image_paths: List[str] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        if not os.path.isdir(self.images_dir):
            return
        for name in os.listdir(self.images_dir):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(self.images_dir, name))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, base + '.txt')

        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes: List[List[float]] = []
        labels: List[int] = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls, x_c, y_c, bw, bh = parts
                        cls_id = int(float(cls))
                        # Only face class (0) supported
                        if cls_id != 0:
                            continue
                        x_center = float(x_c) * w
                        y_center = float(y_c) * h
                        box_w = float(bw) * w
                        box_h = float(bh) * h
                        x1 = max(0.0, x_center - box_w / 2.0)
                        y1 = max(0.0, y_center - box_h / 2.0)
                        x2 = min(float(w), x_center + box_w / 2.0)
                        y2 = min(float(h), y_center + box_h / 2.0)
                        # torchvision detection expects [x1,y1,x2,y2]
                        boxes.append([x1, y1, x2, y2])
                        # label 1 for face (0 is background)
                        labels.append(1)
            except Exception:
                pass

        # Convert to tensors for torchvision detection API
        if F is not None:
            img_tensor = F.to_tensor(img)
        else:
            arr = np.array(img)
            img_tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx], dtype=torch.int64),
        }

        return img_tensor, target


def detection_collate_fn(batch):
    """Custom collate function for detection DataLoader (handles variable-size targets)."""
    images, targets = list(zip(*batch))
    return list(images), list(targets)


