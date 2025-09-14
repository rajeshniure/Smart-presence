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
_embedding_index: Optional[Dict[int, Dict[str, object]]] = None



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
    global _mtcnn, _resnet, _embedding_index
    with _models_lock:
        _mtcnn = None
        _resnet = None
        _embedding_index = None
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


