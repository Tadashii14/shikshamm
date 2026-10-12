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
    _select_primary_face,
    compute_face_embedding,
    cosine_similarity,
    face_match_margin,
    face_match_threshold,
)
from app.routers import face as face_router
from fastapi.testclient import TestClient
from app.main import app
from app.db import init_db


init_db()
client = TestClient(app)


def camera_frame_files(content=b"test-frame"):
    return [("file", (f"frame-{index}.jpg", content, "image/jpeg")) for index in range(5)]

class TestFaceRecognition:
    @pytest.mark.parametrize(
        ("backend", "expected_code"),
        [("opencv-lbp", "no_face_detected"), (None, "face_backend_unavailable")],
    )
    def test_auto_identify_distinguishes_no_face_from_missing_backend(
        self, monkeypatch, backend, expected_code
    ):
        monkeypatch.setattr(face_router, "compute_face_embedding", lambda _data: None)
        monkeypatch.setattr(face_router, "get_face_backend", lambda: backend)

        response = client.post(
            "/face/auto-identify-and-mark",
            data={"code": "camera-test"},
            files=camera_frame_files(b"not-needed-by-mock"),
        )

        assert response.status_code == 200
        assert response.json()["error_code"] == expected_code
        if expected_code == "no_face_detected":
            assert "move back" in response.json()["message"]

    def test_auto_identify_rejects_single_frame_requests(self):
        response = client.post(
            "/face/auto-identify-and-mark",
            data={"code": "camera-test"},
            files={"file": ("frame.jpg", b"frame", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "five camera frames" in response.json()["detail"]

    def test_auto_identify_skips_ambiguous_frames_when_no_clear_consensus(self, monkeypatch):
        from fastapi import HTTPException
        from types import SimpleNamespace

        monkeypatch.setattr(face_router, "_rank_face_users", lambda _embedding, _users: [(1, 0.90), (2, 0.89)])
        embeddings = [np.zeros(512, dtype=np.float32) for _ in range(5)]
        with pytest.raises(HTTPException, match="consistently") as error:
            face_router._identify_consistent_user(embeddings, [SimpleNamespace(id=1), SimpleNamespace(id=2)])
        assert error.value.status_code == 401

    def test_auto_identify_uses_majority_when_one_or_two_frames_flicker(self, monkeypatch):
        from fastapi import HTTPException
        from types import SimpleNamespace

        monkeypatch.setattr(
            face_router,
            "_rank_face_users",
            lambda embedding, _users: [(1, 0.90), (2, 0.80)] if embedding[0] == 0 else [(2, 0.90), (1, 0.80)],
        )
        embeddings = [np.array([index % 2], dtype=np.float32) for index in range(5)]
        user_id, similarity = face_router._identify_consistent_user(
            embeddings, [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        )
        assert user_id == 1
        assert similarity == 0.90

    def test_auto_identify_rejects_frames_without_a_majority(self, monkeypatch):
        from fastapi import HTTPException
        from types import SimpleNamespace

        monkeypatch.setattr(
            face_router,
            "_rank_face_users",
            lambda embedding, _users: [(int(embedding[0]), 0.90), (99, 0.70)],
        )
        embeddings = [np.array([index], dtype=np.float32) for index in [1, 2, 1, 2, 3]]
        with pytest.raises(HTTPException, match="unclear across frames") as error:
            face_router._identify_consistent_user(
                embeddings, [SimpleNamespace(id=1), SimpleNamespace(id=2), SimpleNamespace(id=3)]
            )
        assert error.value.status_code == 401

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

    def test_cosine_similarity_treats_corrupt_embeddings_as_non_matches(self):
        assert cosine_similarity(np.array([np.nan, 1.0]), np.array([1.0, 0.0])) == 0.0
        assert cosine_similarity(np.array([np.inf, 1.0]), np.array([1.0, 0.0])) == 0.0
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0])) == 0.0

    def test_face_ranking_skips_corrupt_and_incompatible_profiles(self):
        from types import SimpleNamespace

        users = [
            SimpleNamespace(id=1, face_embedding=[1.0, 0.0]),
            SimpleNamespace(id=2, face_embedding=[float("nan"), 1.0]),
            SimpleNamespace(id=3, face_embedding=[1.0]),
        ]
        ranked = face_router._rank_face_users(np.array([1.0, 0.0]), users)
        assert ranked == [(1, 1.0)]

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
        assert face_match_margin(embedding) == 0.01
        assert face_match_threshold([0.1] * 512) == 0.35
        assert face_match_margin([0.1] * 512) == 0.06

    def test_primary_face_ignores_small_background_faces_but_rejects_equal_faces(self):
        # One close subject plus a much smaller background/partial face.
        assert _select_primary_face([(20, 10, 260, 300), (410, 80, 90, 100)]) == (20, 10, 260, 300)
        # Two similarly sized people are still ambiguous and fail closed.
        assert _select_primary_face([(20, 10, 180, 200), (240, 10, 170, 190)]) is None
        assert _select_primary_face([]) is None

    def test_consistent_lbp_match_uses_calibrated_small_margin(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(face_router, "_rank_face_users", lambda _embedding, _users: [(1, 0.90), (2, 0.88)])
        embeddings = [np.zeros(4096, dtype=np.float32) for _ in range(5)]
        user_id, similarity = face_router._identify_consistent_user(
            embeddings, [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        )
        assert user_id == 1
        assert similarity == 0.90

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
            files=camera_frame_files(),
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
            files=camera_frame_files(),
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
            files=camera_frame_files(),
        )
        assert identified.status_code == 200
        assert identified.json()["user_id"] == student_id
        assert identified.json()["student_name"] == "Face Test Student"
        assert identified.json()["message"] == "Marked present"


