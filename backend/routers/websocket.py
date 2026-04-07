from fastapi import APIRouter, WebSocket
from backend.services.ast_core import generate
import json

router = APIRouter()

@router.websocket("/ws/code")
async def ws_code(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data = await ws.receive_text()
            payload = json.loads(data)
            nodes = payload.get("nodes", [])
            edges = payload.get("edges", [])
            code = generate({"nodes": nodes, "edges": edges})
            await ws.send_text(json.dumps({"code": code}))
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
