from __future__ import annotations

import io
import math
from functools import lru_cache
from typing import Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore


def _lazy_imports():
    try:
        import onnxruntime as ort  # type: ignore
        import insightface  # type: ignore
        import cv2  # type: ignore
        print("Face recognition dependencies loaded successfully!")
        return ort, insightface, cv2
    except ImportError as e:
        print(f"Face recognition dependencies not available: {e}")
        return None, None, None


@lru_cache(maxsize=1)
def get_insightface_model():
    ort, insightface, _cv2 = _lazy_imports()
    if ort is None or insightface is None:
        print("Face recognition model not available - dependencies missing")
        return None
    try:
        providers = ["CPUExecutionProvider"]
        model = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        model.prepare(ctx_id=0, det_size=(320, 320))  # Smaller detection size for speed
        print("InsightFace model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load InsightFace model: {e}")
        return None


def _opencv_lbp_embedding(image):
    """Return a CPU-only LBP face descriptor when InsightFace is unavailable.

    This classic local descriptor is suitable for a student-project demo. It is
    less robust than InsightFace and is not a liveness or anti-spoofing check.
    """
    try:
        import cv2  # type: ignore

        if np is None:
            return None
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48)
        )
        # Low contrast is common with USB webcams. Retry after equalization.
        if len(faces) == 0:
            faces = detector.detectMultiScale(
                cv2.equalizeHist(gray), scaleFactor=1.08, minNeighbors=4, minSize=(48, 48)
            )
        primary = _select_primary_face(faces)
        if primary is None:
            return None
        x, y, width, height = primary
        face = cv2.equalizeHist(gray[y : y + height, x : x + width])
        face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        return _lbp_descriptor(face)
    except Exception as exc:
        print(f"OpenCV LBP face processing failed ({exc.__class__.__name__})")
        return None


def _select_primary_face(faces):
    """Select a clearly dominant foreground face; reject similarly sized faces.

    Classroom backgrounds often contain small/partial faces. Requiring exactly
    one Haar box makes attendance fail whenever anybody is behind the student.
    A face must be at least 1.75x the area of the next detection to be selected;
    close-sized detections remain ambiguous and fail closed.
    """
    try:
        boxes = [tuple(map(int, box)) for box in faces]
        if not boxes:
            return None
        boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        if len(boxes) > 1:
            largest_area = boxes[0][2] * boxes[0][3]
            second_area = boxes[1][2] * boxes[1][3]
            if second_area <= 0 or largest_area < 1.75 * second_area:
                return None
        return boxes[0]
    except (TypeError, ValueError, IndexError):
        return None


def _lbp_descriptor(face_gray):
    """Build a normalized 4x4-grid LBP histogram from a grayscale face crop."""
    if np is None:
        return None
    center = face_gray[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    neighbors = (
        face_gray[:-2, :-2], face_gray[:-2, 1:-1], face_gray[:-2, 2:], face_gray[1:-1, 2:],
        face_gray[2:, 2:], face_gray[2:, 1:-1], face_gray[2:, :-2], face_gray[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        code |= (neighbor >= center).astype(np.uint8) << bit

    descriptors = []
    for row in np.array_split(code, 4, axis=0):
        for cell in np.array_split(row, 4, axis=1):
            histogram = np.histogram(cell, bins=256, range=(0, 256))[0].astype(np.float32)
            histogram /= max(float(histogram.sum()), 1.0)
            descriptors.append(np.sqrt(histogram))
    vector = np.concatenate(descriptors).astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


def get_face_backend() -> str | None:
    """Identify the real, loaded recognition backend without claiming proxy mode."""
    if get_insightface_model() is not None:
        return "insightface"
    if np is None:
        return None
    try:
        import cv2  # type: ignore

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return "opencv-lbp" if not detector.empty() else None
    except Exception:
        return None


def face_match_threshold(embedding) -> float:
    """Use a threshold calibrated for the stored embedding type."""
    try:
        return 0.82 if len(embedding) == 4096 else 0.35
    except Exception:
        return 0.35


def face_match_margin(embedding) -> float:
    """Require more separation from the weaker LBP fallback than neural embeddings can provide."""
    try:
        return 0.01 if len(embedding) == 4096 else 0.06
    except Exception:
        return 0.06


def _read_image_from_bytes(image_bytes: bytes):
    try:
        import cv2  # type: ignore
        if np is None:
            return None
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return None
        # Resize image for faster processing
        height, width = image.shape[:2]
        if height > 480 or width > 640:
            scale = min(480/height, 640/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        return image
    except Exception:
        return None


def compute_face_embedding(image_bytes: bytes):
    try:
        image = _read_image_from_bytes(image_bytes)
        if image is None:
            return None
        model = get_insightface_model()
        if model is not None:
            faces = model.get(image)
            if len(faces) != 1:
                return None
            face = faces[0]
            emb = face.normed_embedding
            if emb is None or np is None:
                return None
            return np.array(emb, dtype=np.float32)
        return _opencv_lbp_embedding(image)
    except Exception as e:
        print(f"Face embedding computation failed: {e}")
        return None


def cosine_similarity(a, b) -> float:
    if np is None:
        # Simple pure-Python cosine
        def norm(v):
            return sum(x*x for x in v) ** 0.5
        try:
            if len(a) != len(b) or not a or any(not math.isfinite(float(x)) for x in a) or any(not math.isfinite(float(x)) for x in b):
                return 0.0
        except (TypeError, ValueError):
            return 0.0
        na = norm(a) + 1e-8
        nb = norm(b) + 1e-8
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        similarity = float(dot / (na * nb))
        return similarity if math.isfinite(similarity) else 0.0
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if a.size != b.size or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    similarity = float(np.dot(a_norm, b_norm))
    return similarity if math.isfinite(similarity) else 0.0


