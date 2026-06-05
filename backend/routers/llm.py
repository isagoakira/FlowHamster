import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.schemas.llm_request import ApiKeyRequest, ChatCompletionRequest
from backend.schemas.llm_response import ApiKeyStatusResponse, ChatCompletionResponse, LLMModelInfo, LLMProviderInfo, LLMProvidersResponse
from backend.services.llm_adapters import (
    EncryptedApiKeyStore,
    ProviderConfigLoader,
    adapter_for,
    resolve_api_key,
)

router = APIRouter()

CHAT_COMPLETIONS_ROUTE = "/llm/chat/completions"
CHAT_COMPLETIONS_COMPAT_KEY = "chat.completions"

config_loader = ProviderConfigLoader()
api_key_store = EncryptedApiKeyStore()


def _provider_or_404(provider_id: str):
    provider = config_loader.get(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown LLM provider '{provider_id}'")
    return provider


@router.get("/llm/providers", response_model=LLMProvidersResponse)
async def list_providers():
    providers = [
        LLMProviderInfo(
            id=provider.id,
            name=provider.name,
            models=[LLMModelInfo(id=m, name=m) for m in provider.models],
            default_model=provider.default_model,
            enabled=provider.enabled,
            supports_stream=provider.supports_stream,
            supports_tools=provider.supports_tools,
            api_key_configured=api_key_store.configured(provider),
        )
        for provider in config_loader.providers()
    ]
    return LLMProvidersResponse(providers=providers)


@router.put("/llm/providers/{provider_id}/api-key", response_model=ApiKeyStatusResponse)
async def set_api_key(provider_id: str, request: ApiKeyRequest):
    _provider_or_404(provider_id)
    api_key_store.set(provider_id, request.api_key)
    return ApiKeyStatusResponse(provider=provider_id, api_key_configured=True)


@router.delete("/llm/providers/{provider_id}/api-key", response_model=ApiKeyStatusResponse)
async def delete_api_key(provider_id: str):
    provider = _provider_or_404(provider_id)
    api_key_store.delete(provider_id)
    return ApiKeyStatusResponse(provider=provider_id, api_key_configured=api_key_store.configured(provider))


@router.post(CHAT_COMPLETIONS_ROUTE, response_model=ChatCompletionResponse | None)
async def chat_completions(request: ChatCompletionRequest):
    provider = _provider_or_404(request.provider)
    adapter = adapter_for(provider)
    api_key = resolve_api_key(provider, api_key_store, request.api_key)

    if request.stream:
        return StreamingResponse(_stream_completion(adapter, request, api_key), media_type="text/event-stream")

    try:
        return adapter.complete(request, api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        return ChatCompletionResponse(
            success=False,
            provider=provider.id,
            model=request.model or provider.default_model,
            error=str(exc),
        )


def _stream_completion(adapter, request: ChatCompletionRequest, api_key: str | None) -> Iterator[str]:
    try:
        yield from adapter.stream(request, api_key)
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'provider': request.provider, 'error': str(exc)}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'provider': request.provider}, ensure_ascii=False)}\n\n"
