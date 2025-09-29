"""
Unit tests for authentication module
"""
import pytest
from datetime import datetime, timedelta
from app.auth import (
    verify_password, get_password_hash, create_access_token, 
    decode_token, verify_token, refresh_access_token, create_refresh_token
)

class TestPasswordHashing:
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)

class TestJWTTokens:
    def test_create_access_token(self):
        """Test access token creation"""
        user_id = "test_user_123"
        token = create_access_token(user_id)
        
        assert token is not None
        assert isinstance(token, str)
        
        # Decode and verify
        decoded_user = decode_token(token)
        assert decoded_user == user_id

    def test_create_refresh_token(self):
        """Test refresh token creation"""
        user_id = "test_user_123"
        token = create_refresh_token(user_id)
        
        assert token is not None
        assert isinstance(token, str)
        
        # Verify token type
        payload = verify_token(token, "refresh")
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"

    def test_token_expiration(self):
        """Test token expiration"""
        user_id = "test_user_123"
        # Create token with very short expiration
        token = create_access_token(user_id, expires_delta=timedelta(seconds=-1))
        
        # Should be expired
        decoded_user = decode_token(token)
        assert decoded_user is None

    def test_invalid_token(self):
        """Test invalid token handling"""
        invalid_token = "invalid.token.here"
        decoded_user = decode_token(invalid_token)
        assert decoded_user is None

    def test_refresh_token_flow(self):
        """Test refresh token flow"""
        user_id = "test_user_123"
        refresh_token = create_refresh_token(user_id)
        
        # Use refresh token to get new access token
        new_access_token = refresh_access_token(refresh_token)
        assert new_access_token is not None
        
        # Verify new access token
        decoded_user = decode_token(new_access_token)
        assert decoded_user == user_id

    def test_wrong_token_type(self):
        """Test using wrong token type"""
        user_id = "test_user_123"
        access_token = create_access_token(user_id)
        
        # Try to use access token as refresh token
        payload = verify_token(access_token, "refresh")
        assert payload is None
