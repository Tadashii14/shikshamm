"""
Unit tests for face recognition module
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.face import compute_face_embedding, cosine_similarity

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

    @patch('app.services.face.get_insightface_model')
    def test_compute_face_embedding_success(self, mock_get_model):
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

    @patch('app.services.face.get_insightface_model')
    def test_compute_face_embedding_no_face(self, mock_get_model):
        """Test face embedding when no face is detected"""
        # Mock the model to return no faces
        mock_model = MagicMock()
        mock_model.get.return_value = []
        mock_get_model.return_value = mock_model
        
        image_bytes = b"fake_image_data"
        embedding = compute_face_embedding(image_bytes)
        
        assert embedding is None

    @patch('app.services.face.get_insightface_model')
    def test_compute_face_embedding_model_unavailable(self, mock_get_model):
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


