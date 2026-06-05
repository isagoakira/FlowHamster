from typing import Any

from pydantic import BaseModel


class LLMModelInfo(BaseModel):
    id: str
    name: str


class LLMProviderInfo(BaseModel):
    id: str
    name: str
    models: list[LLMModelInfo]
    default_model: str | None = None
    enabled: bool = True
    supports_stream: bool = True
    supports_tools: bool = False
    api_key_configured: bool = False


class LLMProvidersResponse(BaseModel):
    providers: list[LLMProviderInfo]


class ApiKeyStatusResponse(BaseModel):
    provider: str
    api_key_configured: bool


class ChatCompletionResponse(BaseModel):
    success: bool
    provider: str
    model: str | None = None
    message: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    raw: dict[str, Any] | str | None = None
