"""
Rate limiting middleware using slowapi
"""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
import structlog

logger = structlog.get_logger()

# Create limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000/hour", "100/minute"]
)

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler"""
    logger.warning(f"Rate limit exceeded for {request.client.host}: {exc.detail}")
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded: {exc.detail}. Please try again later."
    )

# Set custom handler
limiter._rate_limit_exceeded_handler = rate_limit_exceeded_handler

# Rate limit decorators for different endpoints
def face_recognition_limit():
    """Rate limit for face recognition endpoints"""
    return limiter.limit("10/minute")

def ai_generation_limit():
    """Rate limit for AI generation endpoints"""
    return limiter.limit("5/minute")

def general_api_limit():
    """Rate limit for general API endpoints"""
    return limiter.limit("60/minute")

def admin_limit():
    """Rate limit for admin endpoints"""
    return limiter.limit("200/minute")


