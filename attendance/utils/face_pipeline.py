from typing import List, Dict, Optional, Tuple
import threading
import torch


# Optional fallback to OpenCV if facenet-pytorch is unavailable
_use_fallback = False
try:
    from PIL import Image
    import numpy as np
    from facenet_pytorch import MTCNN, InceptionResnetV1
except Exception:  # pragma: no cover - fallback path
    _use_fallback = True
    import cv2  # type: ignore
    import numpy as np  # type: ignore


_models_lock = threading.Lock()
_mtcnn: Optional["MTCNN"] = None
_resnet: Optional["InceptionResnetV1"] = None
_device: Optional["torch.device"] = None
_embedding_index: Optional[Dict[int, Dict[str, object]]] = None  # student_id -> {name, roll, emb}
_use_custom_trained = False
_custom_recog_model = None
_custom_recog_classes: Optional[List[str]] = None
_custom_recog_arch: Optional[str] = None
_custom_detect_model = None
_custom_detect_is_frcnn = False


def _load_models_if_needed() -> None:
    global _mtcnn, _resnet, _device
    if _use_fallback:
        return
    if _mtcnn is not None and _resnet is not None:
        return
    with _models_lock:
        if _mtcnn is not None and _resnet is not None:
            return
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=_device)
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(_device)


def _rebuild_embedding_index() -> None:
    """Build an in-memory embedding index from verified students' profile images."""
    global _embedding_index
    _embedding_index = {}

    # Avoid import at module load to prevent app registry issues
    from django.db.models import Q
    from attendance.models import Student

    students = Student.objects.filter(Q(is_verified=True) & ~Q(image=""))
    if _use_fallback:
        # Fallback: no embeddings, empty index
        for s in students:
            _embedding_index[int(s.id)] = {"name": s.name, "roll": s.roll_number, "emb": None}
        return

    from PIL import Image
    import numpy as np
    _load_models_if_needed()
    assert _mtcnn is not None and _resnet is not None and _device is not None

    for s in students:
        try:
            if not s.image:
                continue
            img = Image.open(s.image.path).convert('RGB')
            # Use MTCNN to get a single aligned face from the reference image
            aligned, prob = _mtcnn(img, return_prob=True)
            if aligned is None:
                continue
            # If multiple faces returned, pick the highest probability
            if aligned.ndim == 4:  # [n, 3, 160, 160]
                if isinstance(prob, (list, tuple)):
                    idx = int(np.argmax(np.array(prob)))
                else:
                    idx = 0
                aligned_face = aligned[idx:idx+1]
            else:  # Single face tensor [3,160,160]
                aligned_face = aligned.unsqueeze(0)
            with torch.no_grad():
                emb = _resnet(aligned_face.to(_device))
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            _embedding_index[int(s.id)] = {
                "name": s.name,
                "roll": s.roll_number,
                "emb": emb.squeeze(0).cpu().numpy().astype(np.float32),
            }
        except Exception:
            continue


def reload_models() -> None:
    """Reload models and rebuild the embedding index (used after retraining)."""
    global _mtcnn, _resnet, _embedding_index, _use_custom_trained, _custom_recog_model, _custom_recog_classes, _custom_detect_model
    with _models_lock:
        _mtcnn = None
        _resnet = None
        _embedding_index = None
        _use_custom_trained = False
        _custom_recog_model = None
        _custom_recog_classes = None
        _custom_detect_model = None
    if not _use_fallback:
        _load_models_if_needed()
    _rebuild_embedding_index()
    _try_load_custom_models()


def _try_load_custom_models() -> None:
    """Attempt to load custom trained CNN models for detection and recognition."""
    import os
    import torch
    from attendance.utils.custom_models import FaceRecognitionCNN, FaceDetectionCNN
    from torchvision.models import resnet18
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    models_dir = os.path.join(__import__('pathlib').Path(__file__).resolve().parents[1], 'models')
    rec_path = os.path.join(models_dir, 'best_recognition_model.pth')
    det_path = os.path.join(models_dir, 'best_detection_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if os.path.exists(rec_path):
            ckpt = torch.load(rec_path, map_location=device)
            class_names = ckpt.get('class_names')
            arch = ckpt.get('arch')
            if class_names:
                if arch == 'resnet18':
                    model = resnet18(weights=None)
                    in_features = model.fc.in_features
                    model.fc = __import__('torch').nn.Linear(in_features, len(class_names))
                    model.load_state_dict(ckpt['state_dict'])
                    model = model.to(device).eval()
                    globals()['_custom_recog_arch'] = 'resnet18'
                else:
                    model = FaceRecognitionCNN(num_classes=len(class_names)).to(device)
                    model.load_state_dict(ckpt['state_dict'])
                    model.eval()
                    globals()['_custom_recog_arch'] = 'custom_cnn'
                globals()['_custom_recog_model'] = model
                globals()['_custom_recog_classes'] = class_names
                globals()['_use_custom_trained'] = True
    except Exception:
        pass
    try:
        if os.path.exists(det_path):
            # Try to load as Faster R-CNN first (new pipeline)
            try:
                frcnn = fasterrcnn_resnet50_fpn(weights='DEFAULT')
                in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
                frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
                frcnn.load_state_dict(torch.load(det_path, map_location=device))
                frcnn.eval().to(device)
                globals()['_custom_detect_model'] = frcnn
                globals()['_custom_detect_is_frcnn'] = True
            except Exception:
                # Fallback to legacy custom CNN detector
                model = FaceDetectionCNN().to(device)
                state = torch.load(det_path, map_location=device)
                model.load_state_dict(state)
                model.eval()
                globals()['_custom_detect_model'] = model
                globals()['_custom_detect_is_frcnn'] = False
    except Exception:
        pass


def _ensure_ready() -> None:
    if _use_fallback:
        global _fallback_detector
        if '_fallback_detector' not in globals():
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            globals()['_fallback_detector'] = cv2.CascadeClassifier(cascade_path)
        return
    _load_models_if_needed()
    if _embedding_index is None:
        _rebuild_embedding_index()


def _match_embedding(face_emb: "np.ndarray", threshold: float) -> Tuple[str, float]:
    """Return (name, distance_confidence) for the closest match or ("Unknown", 0.0)."""
    assert _embedding_index is not None
    if not _embedding_index:
        return "Unknown", 0.0
    import numpy as np
    ids = list(_embedding_index.keys())
    embs = [(_embedding_index[i]["emb"]) for i in ids if _embedding_index[i]["emb"] is not None]
    meta = [(_embedding_index[i]["name"]) for i in ids if _embedding_index[i]["emb"] is not None]
    if len(embs) == 0:
        return "Unknown", 0.0
    embs_mat = np.stack(embs, axis=0)  # [m, 512]
    dists = np.linalg.norm(embs_mat - face_emb[None, :], axis=1)
    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])
    if min_dist <= threshold:
        # Map distance to confidence [0,1], higher is better
        confidence = max(0.0, 1.0 - (min_dist / max(threshold, 1e-6)))
        return str(meta[min_idx]), confidence
    return "Unknown", 0.0


def detect_and_recognize(image_path: str) -> List[Dict]:
    """Detect faces and recognize using FaceNet embeddings.

    Returns a list of dicts like:
    {"name": str, "box": [x, y, w, h], "confidence": float}
    """
    _ensure_ready()

    # Get recognition threshold from settings
    try:
        from attendance.models import AttendanceSettings
        settings = AttendanceSettings.get_settings()
        threshold = float(settings.face_recognition_threshold)
    except Exception:
        threshold = 0.9

    results: List[Dict] = []

    if _use_fallback:
        img = cv2.imread(image_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = globals()['_fallback_detector'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            results.append({"name": "Unknown", "box": [int(x), int(y), int(w), int(h)], "confidence": 0.0})
        return results

    # If custom trained models are available, use them
    if globals().get('_use_custom_trained') and globals().get('_custom_recog_model') is not None:
        import cv2
        from torchvision.transforms import functional as F
        from attendance.utils.custom_models import preprocess_image_to_tensor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get boxes from detector
        boxes: List[List[int]] = []
        det_model = globals().get('_custom_detect_model')
        if det_model is not None and globals().get('_custom_detect_is_frcnn'):
            img_tensor = F.to_tensor(img_rgb).to(device)
            with torch.no_grad():
                pred = det_model([img_tensor])[0]
            pboxes = pred.get('boxes')
            pscores = pred.get('scores')
            if pboxes is not None and pscores is not None:
                keep = (pscores.detach().cpu().numpy() >= 0.5)
                for b, k in zip(pboxes.detach().cpu().numpy(), keep):
                    if not k:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in b]
                    boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
        elif det_model is not None:
            # Legacy sliding window on custom CNN
            from attendance.utils.custom_models import sliding_window_detect
            boxes = sliding_window_detect(img_rgb, det_model, device)
        else:
            # Fallback to MTCNN for boxes
            from PIL import Image
            img_pil = Image.fromarray(img_rgb)
            boxes_mtcnn, _ = _mtcnn.detect(img_pil)
            if boxes_mtcnn is not None:
                for b in boxes_mtcnn:
                    x1, y1, x2, y2 = [int(v) for v in b]
                    boxes.append([x1, y1, int(x2 - x1), int(y2 - y1)])

        # Recognize each box using custom recognition CNN
        recog_model = globals()['_custom_recog_model']
        class_names = globals()['_custom_recog_classes'] or []
        for (x, y, w, h) in boxes:
            x2, y2 = x + w, y + h
            crop = img_rgb[max(0, y):max(0, y2), max(0, x):max(0, x2)]
            if crop.size == 0:
                continue
            tensor = preprocess_image_to_tensor(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = recog_model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                name = class_names[idx] if 0 <= idx < len(class_names) else 'Unknown'
                conf = float(probs[idx])
            results.append({"name": name, "box": [int(x), int(y), int(w), int(h)], "confidence": conf})
        return results

    # facenet-pytorch path
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert('RGB')

    assert _mtcnn is not None and _resnet is not None and _device is not None
    boxes, probs = _mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    # Filter by detection probability
    filtered: List[Tuple[np.ndarray, float]] = []
    for b, p in zip(boxes, probs):
        if p is None:
            continue
        if float(p) >= 0.85:
            filtered.append((b, float(p)))
    if not filtered:
        return []

    aligned = _mtcnn.extract(img, np.stack([b for b, _ in filtered], axis=0), None)
    with torch.no_grad():
        embs = _resnet(aligned.to(_device))
        embs = torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy().astype(np.float32)

    for (box, det_p), emb in zip(filtered, embs):
        name, conf = _match_embedding(emb, threshold)
        x1, y1, x2, y2 = [int(v) for v in box]
        results.append({
            "name": name,
            "box": [x1, y1, int(x2 - x1), int(y2 - y1)],
            "confidence": conf,
        })

    return results


