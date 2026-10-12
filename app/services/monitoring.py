from __future__ import annotations
"""
Health checks and monitoring with Prometheus metrics
"""
from fastapi import HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import psutil
import structlog
from app.services.cache import cache
from app import db as db_module
from app.models import User, AttendanceSession
from sqlmodel import Session, select

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_SESSIONS = Gauge('active_sessions_total', 'Number of active attendance sessions')
TOTAL_USERS = Gauge('total_users', 'Total number of users in database')
FACE_RECOGNITION_REQUESTS = Counter('face_recognition_requests_total', 'Total face recognition requests', ['status'])
AI_GENERATION_REQUESTS = Counter('ai_generation_requests_total', 'Total AI generation requests', ['type', 'status'])

class HealthChecker:
    def __init__(self):
        self.start_time = time.time()

    def check_database(self) -> dict:
        """Check database connectivity and health"""
        # If startup already determined a configured database is unreachable,
        # report that honestly (without attempting a query that could hang or
        # echo the connection string). See app.db.DB_AVAILABLE / DB_ERROR.
        if not db_module.DB_AVAILABLE:
            return {
                "status": "unhealthy",
                "message": db_module.DB_ERROR or "Database unavailable",
            }
        try:
            if db_module.engine is None:
                raise RuntimeError("database engine is unavailable")
            with Session(db_module.engine) as session:
                # Only check that the table is readable; do not load every user.
                users = session.exec(select(User).limit(1)).all()

            return {
                "status": "healthy",
                "message": "Database connection successful",
                "users_count": len(users)
            }
        except Exception as e:
            # Never echo the raw exception text: a driver error can embed the
            # DSN (including credentials). Log only the exception type.
            db_module.mark_database_unavailable(e, "health check")
            logger.error("Database health check failed", error_type=e.__class__.__name__)
            return {
                "status": "unhealthy",
                "message": db_module.DB_ERROR or "Database connection failed",
            }

    def check_cache(self) -> dict:
        """Check cache connectivity"""
        try:
            # Test cache operations
            test_key = "health_check_test"
            cache.set(test_key, "test_value", expire=10)
            value = cache.get(test_key)
            cache.delete(test_key)
            
            if value == "test_value":
                return {
                    "status": "healthy",
                    "message": "Cache operations successful"
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Cache operations failed"
                }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Cache connection failed: {str(e)}"
            }

    def check_face_recognition(self) -> dict:
        """Check face recognition system"""
        try:
            from app.services.face import get_face_backend
            backend = get_face_backend()

            if backend == "insightface":
                return {
                    "status": "healthy",
                    "message": "InsightFace recognition model loaded successfully"
                }
            if backend == "opencv-lbp":
                return {
                    "status": "healthy",
                    "message": "OpenCV LBP face matching is ready for local demo attendance; no liveness check is provided."
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Face matching is unavailable. Install the optional local camera dependencies from requirements-local.txt."
                }
        except Exception as e:
            logger.error(f"Face recognition health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Face recognition failed: {str(e)}"
            }

    def get_system_metrics(self) -> dict:
        """Get system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "uptime_seconds": time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return {"error": str(e)}

    def get_application_metrics(self) -> dict:
        """Get application-specific metrics"""
        if not db_module.DB_AVAILABLE or db_module.engine is None:
            return {"error": "Database unavailable"}
        try:
            with Session(db_module.engine) as session:
                active_sessions = session.exec(
                    select(AttendanceSession).where(AttendanceSession.is_active == True)
                ).all()
                total_users = session.exec(select(User)).all()
            
            # Update Prometheus gauges
            ACTIVE_SESSIONS.set(len(active_sessions))
            TOTAL_USERS.set(len(total_users))
            
            return {
                "active_sessions": len(active_sessions),
                "total_users": len(total_users),
                "cache_available": cache.redis_client is not None
            }
        except Exception as e:
            db_module.mark_database_unavailable(e, "application metrics query")
            logger.error("Application metrics collection failed", error_type=e.__class__.__name__)
            return {"error": "Database unavailable"}

    def get_health_status(self) -> dict:
        """Get overall health status"""
        checks = {
            "database": self.check_database(),
            "cache": self.check_cache(),
            "face_recognition": self.check_face_recognition()
        }
        
        system_metrics = self.get_system_metrics()
        app_metrics = self.get_application_metrics()
        
        # Determine overall status
        unhealthy_checks = [name for name, check in checks.items() if check["status"] == "unhealthy"]
        overall_status = "healthy" if not unhealthy_checks else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": checks,
            "system_metrics": system_metrics,
            "application_metrics": app_metrics,
            "unhealthy_components": unhealthy_checks
        }

# Global health checker instance
health_checker = HealthChecker()

def get_metrics():
    """Get Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
