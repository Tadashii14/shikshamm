from __future__ import annotations

import io
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np


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


def _read_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
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


def compute_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        image = _read_image_from_bytes(image_bytes)
        if image is None:
            return None
        model = get_insightface_model()
        if model is None:
            print("Face recognition model not available")
            return None
        faces = model.get(image)
        if not faces:
            return None
        # Use the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding
        if emb is None:
            return None
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        print(f"Face embedding computation failed: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


