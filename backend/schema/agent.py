from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgentMode = Literal["react", "cot"]
AgentRole = Literal["system", "user", "assistant", "tool"]
ToolRisk = Literal["low", "medium", "high"]


class AgentGraphContext(BaseModel):
    model_graph: dict[str, Any] | None = None
    data_graph: dict[str, Any] | None = None
    bindings: list[dict[str, Any]] | None = None
    training_config: dict[str, Any] | None = None


class AgentMessage(BaseModel):
    id: int | None = None
    session_id: str
    role: AgentRole
    content: str
    created_at: float
    tool_name: str | None = None


class AgentSessionInfo(BaseModel):
    id: str
    title: str
    summary: str = ""
    created_at: float
    updated_at: float
    message_count: int = 0


class AgentToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    risk: ToolRisk = "low"
    requires_confirmation: bool = False
    confirmation_token: str | None = None


class AgentObservation(BaseModel):
    tool_call_id: str
    tool_name: str
    success: bool
    output: dict[str, Any] | str | None = None
    error: str | None = None


class AgentChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    mode: AgentMode = "react"
    graph_context: AgentGraphContext | None = None
    confirm_token: str | None = None
    max_context_messages: int | None = None


class AgentChatResponse(BaseModel):
    success: bool
    session_id: str
    reply: str
    mode: AgentMode
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    observations: list[AgentObservation] = Field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_token: str | None = None
    blocked: bool = False
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""


class AgentSessionDetail(AgentSessionInfo):
    messages: list[AgentMessage] = Field(default_factory=list)
