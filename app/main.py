from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import structlog

from app.db import init_db
from app.routers import auth as auth_router
from app.routers import admin as admin_router
from app.routers import student as student_router
from app.routers import face as face_router
from app.routers import stats as stats_router
from app.routers import notes as notes_router
from app.routers import advanced as advanced_router
from app.notifications import manager
from app.services.logging import configure_logging, log_api_request
from app.services.monitoring import health_checker, get_metrics, REQUEST_COUNT, REQUEST_DURATION
from app.middleware.rate_limit import limiter

# Configure logging
configure_logging()
logger = structlog.get_logger()

app = FastAPI(
    title="AttendAI",
    description="AI-powered attendance management and smart curriculum system",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Add middleware for request logging and metrics
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # Log request start
    log_api_request(request)
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    # Log request completion
    response.response_time = process_time
    log_api_request(request, response)
    
    return response

# ----------------- Health & Monitoring Endpoints -----------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return health_checker.get_health_status()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return get_metrics()

# ----------------- Pages -----------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request, "title": "AttendAI — Learn smarter"})

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request, "title": "Admin Portal"})

@app.get("/student", response_class=HTMLResponse)
def student_page(request: Request):
    # You can pre-fill student_id from login here if available
    return templates.TemplateResponse("student.html", {"request": request, "title": "Student Portal"})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "title": "Login"})

@app.get("/quiz", response_class=HTMLResponse)
def quiz_page(request: Request):
    return templates.TemplateResponse("quiz.html", {"request": request, "title": "Quiz Generator"})

@app.get("/flashcards", response_class=HTMLResponse)
def flashcards_page(request: Request):
    return templates.TemplateResponse("flashcards.html", {"request": request, "title": "Flashcards Generator"})

@app.get("/timetable", response_class=HTMLResponse)
def timetable_page(request: Request):
    return templates.TemplateResponse("timetable.html", {"request": request, "title": "Study Timetable Generator"})


# ----------------- Startup -----------------
@app.on_event("startup")
def on_startup():
    init_db()


# ----------------- Routers -----------------
app.include_router(auth_router.router)
app.include_router(admin_router.router)
app.include_router(student_router.router)
app.include_router(face_router.router)
app.include_router(stats_router.router)
app.include_router(notes_router.router)
app.include_router(advanced_router.router, prefix="/api/advanced", tags=["Advanced Features"])


# ----------------- WebSocket -----------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    user_id = websocket.query_params.get("user_id")
    if not user_id:
        await websocket.close()
        return
    await manager.connect(user_id, websocket)
    try:
        while True:
            # Keep connection alive; ignore incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user_id, websocket)
