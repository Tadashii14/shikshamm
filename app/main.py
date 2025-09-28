from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.db import init_db
from app.routers import auth as auth_router
from app.routers import admin as admin_router
from app.routers import student as student_router
from app.routers import face as face_router
from app.routers import stats as stats_router
from app.routers import notes as notes_router
from app.notifications import manager

app = FastAPI(title="AttendAI")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# ----------------- Pages -----------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": "AttendAI"})

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
