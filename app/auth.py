from datetime import datetime, timedelta
import os
from typing import Optional
import structlog

from jose import jwt, JWTError
from passlib.context import CryptContext

logger = structlog.get_logger()

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"sub": subject, "exp": expire, "type": "access"}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    logger.info("access_token_created", user_id=subject, expires_at=expire.isoformat())
    return encoded_jwt


def create_refresh_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": subject, "exp": expire, "type": "refresh"}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    logger.info("refresh_token_created", user_id=subject, expires_at=expire.isoformat())
    return encoded_jwt


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != "access":
            logger.warning("invalid_token_type", expected="access", actual=payload.get("type"))
            return None
            
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            logger.warning("token_expired", user_id=payload.get("sub"))
            return None
            
        logger.info("token_verified", user_id=payload.get("sub"))
        return payload.get("sub")
        
    except JWTError as e:
        logger.warning("token_verification_failed", error=str(e))
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[dict]:
    """Verify JWT token and return full payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            logger.warning("invalid_token_type", expected=token_type, actual=payload.get("type"))
            return None
            
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            logger.warning("token_expired", user_id=payload.get("sub"))
            return None
            
        logger.info("token_verified", user_id=payload.get("sub"), token_type=token_type)
        return payload
        
    except JWTError as e:
        logger.warning("token_verification_failed", error=str(e))
        return None


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Refresh access token using refresh token"""
    payload = verify_token(refresh_token, "refresh")
    if not payload:
        return None
        
    # Create new access token
    return create_access_token(payload.get("sub"))


