from typing import List, Dict, Optional, Tuple
import threading
import torch
import os

_use_fallback = False
try:
    from PIL import Image
    import numpy as np
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    _use_fallback = True
    import cv2
    import numpy as np

_models_lock = threading.Lock()
_mtcnn: Optional["MTCNN"] = None
_resnet: Optional["InceptionResnetV1"] = None
_device: Optional["torch.device"] = None
_embedding_index: Optional[Dict[int, Dict[str, object]]] = None

# Custom models
_custom_mobilenet: Optional["tf.keras.Model"] = None
_custom_resnet: Optional["torch.nn.Module"] = None
_custom_models_loaded = False


def _load_custom_models() -> None:
    global _custom_mobilenet, _custom_resnet, _custom_models_loaded, _device
    if _use_fallback or _custom_models_loaded:
        return
    
    try:
        # Load custom MobileNet for face detection
        mobilenet_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'face detection', 'mobilenet_1_0_224_tf.h5')
        if os.path.exists(mobilenet_path):
            _custom_mobilenet = load_model(mobilenet_path)
        
        # Load custom ResNet for face recognition
        resnet_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'face_recognition', 'resnet.pth')
        if os.path.exists(resnet_path):
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(resnet_path, map_location=_device)
            from attendance.trainingScript.Resnet import resnet18
            num_classes = len(checkpoint.get('class_names', []))
            _custom_resnet = resnet18(num_classes)
            _custom_resnet.load_state_dict(checkpoint['state_dict'])
            _custom_resnet.eval().to(_device)
        
        _custom_models_loaded = True
    except Exception:
        _custom_models_loaded = False


def _load_models_if_needed() -> None:
    global _mtcnn, _resnet, _device
    if _use_fallback or (_mtcnn is not None and _resnet is not None):
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

    from django.db.models import Q
    from attendance.models import Student

    students = Student.objects.filter(Q(is_verified=True) & ~Q(image=""))
    if _use_fallback:
        for s in students:
            _embedding_index[int(s.id)] = {"name": s.name, "roll": s.roll_number, "emb": None}
        return

    _load_models_if_needed()
    assert _mtcnn is not None and _resnet is not None and _device is not None

    for s in students:
        try:
            if not s.image:
                continue
            img = Image.open(s.image.path).convert('RGB')
            aligned, prob = _mtcnn(img, return_prob=True)
            if aligned is None:
                continue
            
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
    global _mtcnn, _resnet, _embedding_index, _custom_models_loaded
    with _models_lock:
        _mtcnn = None
        _resnet = None
        _embedding_index = None
        _custom_models_loaded = False
    if not _use_fallback:
        _load_models_if_needed()
    _rebuild_embedding_index()


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


def _detect_faces_custom(image_path: str) -> List[Tuple[np.ndarray, float]]:
    """Detect faces using custom MobileNet model"""
    if _custom_mobilenet is None:
        return []
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = _custom_mobilenet.predict(img_array, verbose=0)
        confidence = float(prediction[0][1])
        
        if confidence > 0.5:
            h, w = img.size[1], img.size[0]
            box = np.array([0, 0, w, h])
            return [(box, confidence)]
        return []
    except Exception:
        return []


def _recognize_face_custom(face_img: np.ndarray) -> Tuple[str, float]:
    """Recognize face using custom ResNet model"""
    if _custom_resnet is None:
        return "Unknown", 0.0
    
    try:
        face_img = Image.fromarray(face_img).resize((224, 224))
        face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(_device)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(_device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(_device)
        face_tensor = (face_tensor - mean) / std
        
        with torch.no_grad():
            outputs = _custom_resnet(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'face_recognition', 'resnet.pth'), map_location='cpu')
            class_names = checkpoint.get('class_names', [])
            
            if predicted.item() < len(class_names):
                return class_names[predicted.item()], confidence.item()
            return "Unknown", 0.0
    except Exception:
        return "Unknown", 0.0


def _match_embedding(face_emb: "np.ndarray", threshold: float) -> Tuple[str, float]:
    """Return (name, distance_confidence) for the closest match or ("Unknown", 0.0)."""
    if not _embedding_index:
        return "Unknown", 0.0
    
    ids = list(_embedding_index.keys())
    embs = [(_embedding_index[i]["emb"]) for i in ids if _embedding_index[i]["emb"] is not None]
    meta = [(_embedding_index[i]["name"]) for i in ids if _embedding_index[i]["emb"] is not None]
    
    if len(embs) == 0:
        return "Unknown", 0.0
    
    embs_mat = np.stack(embs, axis=0)
    dists = np.linalg.norm(embs_mat - face_emb[None, :], axis=1)
    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])
    
    if min_dist <= threshold:
        confidence = max(0.0, 1.0 - (min_dist / max(threshold, 1e-6)))
        return str(meta[min_idx]), confidence
    return "Unknown", 0.0


def detect_and_recognize(image_path: str) -> List[Dict]:
    """Detect faces and recognize using custom models with fallback to MTCNN+InceptionResnetV1."""
    _load_custom_models()
    
    try:
        from attendance.models import AttendanceSettings
        settings = AttendanceSettings.get_settings()
        threshold = float(settings.face_recognition_threshold)
    except Exception:
        threshold = 0.9

    results: List[Dict] = []

    if _custom_models_loaded and _custom_mobilenet is not None and _custom_resnet is not None:
        try:
            detected_faces = _detect_faces_custom(image_path)
            if detected_faces:
                img = Image.open(image_path).convert('RGB')
                img_array = np.array(img)
                
                for box, det_conf in detected_faces:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    face_region = img_array[y1:y2, x1:x2]
                    if face_region.size > 0:
                        name, rec_conf = _recognize_face_custom(face_region)
                        results.append({
                            "name": name,
                            "box": [x1, y1, int(x2 - x1), int(y2 - y1)],
                            "confidence": rec_conf,
                        })
                
                if results:
                    return results
        except Exception:
            pass

    
    _ensure_ready()

    if _use_fallback:
        img = cv2.imread(image_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = globals()['_fallback_detector'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            results.append({"name": "Unknown", "box": [int(x), int(y), int(w), int(h)], "confidence": 0.0})
        return results

    img = Image.open(image_path).convert('RGB')
    assert _mtcnn is not None and _resnet is not None and _device is not None
    
    boxes, probs = _mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    filtered: List[Tuple[np.ndarray, float]] = []
    for b, p in zip(boxes, probs):
        if p is not None and float(p) >= 0.85:
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