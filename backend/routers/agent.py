from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schema.agent import AgentChatRequest, AgentChatResponse, AgentSessionDetail, AgentSessionInfo
from backend.services.agent_orchestrator import AgentOrchestrator

router = APIRouter(prefix="/agent", tags=["agent"])

orchestrator: AgentOrchestrator | None = None


def get_orchestrator() -> AgentOrchestrator:
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator()
    return orchestrator


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest):
    return get_orchestrator().chat(req)


@router.get("/sessions", response_model=list[AgentSessionInfo])
async def list_agent_sessions():
    return get_orchestrator().list_sessions()


@router.get("/sessions/{session_id}", response_model=AgentSessionDetail)
async def get_agent_session(session_id: str):
    session = get_orchestrator().get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Agent session not found")
    return session


@router.delete("/sessions/{session_id}")
async def delete_agent_session(session_id: str):
    if not get_orchestrator().delete_session(session_id):
        raise HTTPException(status_code=404, detail="Agent session not found")
    return {"status": "ok"}
