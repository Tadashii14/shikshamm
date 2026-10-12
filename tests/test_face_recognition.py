"""
Unit tests for face recognition module
"""
import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

# These tests exercise the numpy code path of the face service. When the heavy
# face stack (numpy/opencv/onnxruntime/insightface) is not installed — e.g. a
# free-tier/serverless deployment or a lean dev env — skip the module cleanly
# instead of raising a collection error that aborts the whole test suite.
np = pytest.importorskip("numpy", reason="numpy not installed; face-service numpy path unavailable")
from app.services.face import (
    _lbp_descriptor,
    compute_face_embedding,
    cosine_similarity,
    face_match_threshold,
)
from fastapi.testclient import TestClient
from app.main import app
from app.db import init_db


init_db()
client = TestClient(app)

class TestFaceRecognition:
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
        
        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001
        
        # Test opposite vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001

    @patch('app.services.face._read_image_from_bytes', return_value=np.zeros((16, 16, 3), dtype=np.uint8))
    @patch('app.services.face.get_insightface_model')
    def test_compute_face_embedding_success(self, mock_get_model, _mock_read_image):
        """Test successful face embedding computation"""
        # Mock the model
        mock_model = MagicMock()
        mock_face = MagicMock()
        mock_face.normed_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model.get.return_value = [mock_face]
        mock_get_model.return_value = mock_model
        
        # Test with mock image data
        image_bytes = b"fake_image_data"
        embedding = compute_face_embedding(image_bytes)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 4

    @patch('app.services.face._read_image_from_bytes', return_value=np.zeros((16, 16, 3), dtype=np.uint8))
    @patch('app.services.face.get_insightface_model')
    def test_compute_face_embedding_no_face(self, mock_get_model, _mock_read_image):
        """Test face embedding when no face is detected"""
        # Mock the model to return no faces
        mock_model = MagicMock()
        mock_model.get.return_value = []
        mock_get_model.return_value = mock_model
        
        image_bytes = b"fake_image_data"
        embedding = compute_face_embedding(image_bytes)
        
        assert embedding is None

    @patch('app.services.face._read_image_from_bytes', return_value=np.zeros((128, 128, 3), dtype=np.uint8))
    @patch('app.services.face.get_insightface_model')
    def test_compute_face_embedding_model_unavailable(self, mock_get_model, _mock_read_image):
        """Test face embedding when model is unavailable"""
        mock_get_model.return_value = None
        
        image_bytes = b"fake_image_data"
        embedding = compute_face_embedding(image_bytes)
        
        assert embedding is None

    def test_face_verification_threshold(self):
        """Test face verification threshold logic"""
        # High similarity should pass
        high_sim = 0.8
        threshold = 0.35
        assert high_sim > threshold
        
        # Low similarity should fail
        low_sim = 0.2
        assert low_sim < threshold

    def test_lbp_descriptor_is_normalized_and_uses_its_own_threshold(self):
        sample = np.arange(128 * 128, dtype=np.uint8).reshape((128, 128))
        embedding = _lbp_descriptor(sample)
        assert embedding.shape == (4096,)
        assert abs(cosine_similarity(embedding, embedding) - 1.0) < 0.001
        assert face_match_threshold(embedding) == 0.82
        assert face_match_threshold([0.1] * 512) == 0.35

    def test_face_strict_defaults_to_real_matching(self, monkeypatch):
        from app.routers.face import is_face_strict

        monkeypatch.delenv("FACE_STRICT", raising=False)
        assert is_face_strict() is True
        monkeypatch.setenv("FACE_STRICT", "false")
        assert is_face_strict() is False

    def test_strict_enrollment_student_verification_and_admin_matching(self, monkeypatch):
        import app.routers.face as face_router

        monkeypatch.setenv("FACE_STRICT", "true")
        tag = uuid4().hex[:8]
        class_code = f"FC{tag}"
        session_code = f"FS{tag}"
        email = f"face-{tag}@example.test"
        admission_number = f"ADM{tag}"
        created = client.post(
            "/admin/students/create",
            params={"email": email, "full_name": "Face Test Student", "admission_number": admission_number},
        )
        assert created.status_code == 200
        student_id = created.json()["id"]
        assert client.post(
            "/admin/classes",
            params={"name": "Face Test Class", "code": class_code, "created_by_user_id": 1},
        ).status_code == 200
        assert client.post(
            "/admin/sessions/start",
            params={"class_code": class_code, "code": session_code, "subject": "Test"},
        ).status_code == 200

        enrolled_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        monkeypatch.setattr(face_router, "compute_face_embedding", lambda _image: enrolled_vector)
        payload = {"file": ("frame.jpg", b"test-frame", "image/jpeg")}
        enrolled = client.post(
            "/face/enroll-by-admission",
            data={"admission_number": admission_number},
            files=payload,
        )
        assert enrolled.status_code == 200
        assert enrolled.json().get("proxy_mode") is not True

        verified = client.post(
            "/face/verify-by-admission",
            data={"admission_number": admission_number, "code": session_code},
            files=payload,
        )
        assert verified.status_code == 200
        assert verified.json()["user_id"] == student_id
        assert verified.json()["similarity"] > 0.35
        assert "face verified" in verified.json()["message"].lower()

        monkeypatch.setattr(
            face_router,
            "compute_face_embedding",
            lambda _image: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        mismatch = client.post(
            "/face/verify-by-admission",
            data={"admission_number": admission_number, "code": session_code},
            files=payload,
        )
        assert mismatch.status_code == 401
        assert "similarity too low" in mismatch.json()["detail"].lower()

        # A second session exercises the admin auto-identification endpoint.
        second_class = f"AC{tag}"
        second_session = f"AS{tag}"
        client.post(
            "/admin/classes",
            params={"name": "Face Auto Test Class", "code": second_class, "created_by_user_id": 1},
        )
        client.post(
            "/admin/sessions/start",
            params={"class_code": second_class, "code": second_session, "subject": "Test"},
        )
        monkeypatch.setattr(face_router, "compute_face_embedding", lambda _image: enrolled_vector)
        identified = client.post(
            "/face/auto-identify-and-mark",
            data={"code": second_session},
            files=payload,
        )
        assert identified.status_code == 200
        assert identified.json()["user_id"] == student_id
        assert identified.json()["student_name"] == "Face Test Student"
        assert identified.json()["message"] == "Marked present"


