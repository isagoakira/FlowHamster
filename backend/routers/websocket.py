"""
WebSocket 路由

提供实时双向通信能力：
- /ws/code — 接收 flow JSON，返回生成的代码（流式响应）

WebSocket 与 HTTP 不同，是持久连接，适用于需要即时反馈的场景。
客户端发送 JSON，服务器返回 JSON。
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.code_generation import generate_python
import json

router = APIRouter()


@router.websocket("/ws/code")
async def ws_code(ws: WebSocket):
    """
    WebSocket 实时代码生成端点。

    客户端连接后发送 JSON 消息：
        {"nodes": [...], "edges": [...]}

    服务器返回：
        {"code": "生成的 Python 代码"}
      或
        {"error": "错误信息"}

    连接建立后可持续发送消息，服务器即时响应。
    客户端主动关闭连接时，服务器自动清理。
    """
    await ws.accept()
    while True:
        try:
            data = await ws.receive_text()
            payload = json.loads(data)
            graph = payload.get("graph")
            if graph is None:
                graph = {
                    "nodes": payload.get("nodes", []),
                    "edges": payload.get("edges", []),
                }
            generated = generate_python(
                graph=graph,
                options=payload.get("options"),
                training_config=payload.get("training_config"),
                data_graph=payload.get("data_graph"),
                bindings=payload.get("bindings"),
            )
            response = {"code": generated.code}
            if generated.warnings:
                response["warnings"] = generated.warnings
            await ws.send_text(json.dumps(response))
        except WebSocketDisconnect:
            break
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
