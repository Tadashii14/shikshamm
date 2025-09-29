"""
Comprehensive logging configuration
"""
import structlog
import logging
import sys
from datetime import datetime
import json

def configure_logging():
    """Configure structured logging"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

def get_logger(name: str = None):
    """Get a structured logger"""
    return structlog.get_logger(name)

# Performance logging decorator
def log_performance(func_name: str):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger("performance")
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "function_completed",
                    function=func_name,
                    duration_seconds=duration,
                    status="success"
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    "function_failed",
                    function=func_name,
                    duration_seconds=duration,
                    error=str(e),
                    status="error"
                )
                raise
        return wrapper
    return decorator

# API request logging
def log_api_request(request, response=None, error=None):
    """Log API requests and responses"""
    logger = get_logger("api")
    
    log_data = {
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }
    
    if response:
        log_data.update({
            "status_code": response.status_code,
            "response_time": getattr(response, "response_time", None)
        })
        logger.info("api_request_completed", **log_data)
    elif error:
        log_data.update({
            "error": str(error),
            "status_code": getattr(error, "status_code", 500)
        })
        logger.error("api_request_failed", **log_data)
    else:
        logger.info("api_request_started", **log_data)


