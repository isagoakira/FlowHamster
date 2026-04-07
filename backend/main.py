"""
FlowHamster Backend — FastAPI Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import generate, execute, export, websocket, templates, workflows

app = FastAPI(title="FlowHamster API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://192.168.102.231:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api")
app.include_router(execute.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(websocket.router)
app.include_router(templates.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "FlowHamster API is running", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}
