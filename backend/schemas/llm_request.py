from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    provider: str
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    api_key: str | None = None
    extra: dict[str, Any] | None = None
