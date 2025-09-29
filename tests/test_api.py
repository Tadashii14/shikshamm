"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestHealthEndpoints:
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"

class TestStudentEndpoints:
    def test_student_page(self):
        """Test student page loads"""
        response = client.get("/student")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_admin_page(self):
        """Test admin page loads"""
        response = client.get("/admin")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

class TestFaceEndpoints:
    def test_face_endpoints_exist(self):
        """Test face recognition endpoints exist"""
        # These should return 405 Method Not Allowed for GET
        response = client.get("/face/verify-by-admission")
        assert response.status_code == 405
        
        response = client.get("/face/enroll-by-admission")
        assert response.status_code == 405

class TestAIEndpoints:
    def test_ai_endpoints_exist(self):
        """Test AI generation endpoints exist"""
        # These should return 405 Method Not Allowed for GET
        response = client.get("/api/advanced/pdf/generate-quiz")
        assert response.status_code == 405
        
        response = client.get("/api/advanced/pdf/generate-flashcards")
        assert response.status_code == 405
        
        response = client.get("/api/advanced/timetable/generate")
        assert response.status_code == 405

class TestWebSocket:
    def test_websocket_endpoint(self):
        """Test WebSocket endpoint exists"""
        with client.websocket_connect("/ws?user_id=test") as websocket:
            # Connection should be established
            assert websocket is not None


