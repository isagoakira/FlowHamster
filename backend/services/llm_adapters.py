from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from backend.schemas.llm_request import ChatCompletionRequest, ChatMessage
from backend.schemas.llm_response import ChatCompletionResponse


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_providers.json"


@dataclass(frozen=True)
class LLMProviderConfig:
    id: str
    name: str
    base_url: str
    models: list[str]
    default_model: str | None = None
    api_key_env: str | None = None
    enabled: bool = True
    supports_stream: bool = True
    supports_tools: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMProviderConfig":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            base_url=str(data.get("base_url") or "").rstrip("/"),
            models=list(data.get("models") or []),
            default_model=data.get("default_model"),
            api_key_env=data.get("api_key_env"),
            enabled=bool(data.get("enabled", True)),
            supports_stream=bool(data.get("supports_stream", True)),
            supports_tools=bool(data.get("supports_tools", False)),
            extra=dict(data.get("extra") or {}),
        )


class ProviderConfigLoader:
    def __init__(self, path: str | Path | None = None):
        configured = path or os.environ.get("FLOWHAMSTER_LLM_CONFIG")
        self.path = Path(configured) if configured else DEFAULT_CONFIG_PATH
        self._mtime_ns: int | None = None
        self._providers: dict[str, LLMProviderConfig] = {}

    def providers(self) -> list[LLMProviderConfig]:
        self._reload_if_needed()
        return list(self._providers.values())

    def get(self, provider_id: str) -> LLMProviderConfig | None:
        self._reload_if_needed()
        return self._providers.get(provider_id)

    def _reload_if_needed(self) -> None:
        stat = self.path.stat()
        if self._mtime_ns == stat.st_mtime_ns:
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        providers = payload.get("providers", [])
        self._providers = {
            provider.id: provider
            for provider in (LLMProviderConfig.from_dict(item) for item in providers)
            if provider.enabled
        }
        self._mtime_ns = stat.st_mtime_ns


class EncryptedApiKeyStore:
    def __init__(self, secret: str | None = None):
        seed = secret or os.environ.get("FLOWHAMSTER_KEY_STORE_SECRET") or "flowhamster-local-key-store"
        self._secret = hashlib.sha256(seed.encode("utf-8")).digest()
        self._values: dict[str, str] = {}

    def set(self, provider_id: str, api_key: str) -> None:
        self._values[provider_id] = self._encrypt(api_key)

    def get(self, provider_id: str) -> str | None:
        encrypted = self._values.get(provider_id)
        if not encrypted:
            return None
        return self._decrypt(encrypted)

    def delete(self, provider_id: str) -> None:
        self._values.pop(provider_id, None)

    def configured(self, provider: LLMProviderConfig) -> bool:
        return bool(self._values.get(provider.id) or (provider.api_key_env and os.environ.get(provider.api_key_env)))

    def _encrypt(self, value: str) -> str:
        raw = value.encode("utf-8")
        cipher = bytes(byte ^ self._secret[index % len(self._secret)] for index, byte in enumerate(raw))
        return base64.urlsafe_b64encode(cipher).decode("ascii")

    def _decrypt(self, value: str) -> str:
        cipher = base64.urlsafe_b64decode(value.encode("ascii"))
        raw = bytes(byte ^ self._secret[index % len(self._secret)] for index, byte in enumerate(cipher))
        return raw.decode("utf-8")


class HTTPTransport:
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float = 60,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Provider HTTP {exc.code}: {detail}") from exc

    def stream_json(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float = 60,
    ) -> Iterator[str]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    yield raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Provider HTTP {exc.code}: {detail}") from exc


def resolve_api_key(config: LLMProviderConfig, store: EncryptedApiKeyStore, request_key: str | None = None) -> str | None:
    return request_key or store.get(config.id) or (os.environ.get(config.api_key_env) if config.api_key_env else None)


def _message_dict(message: ChatMessage) -> dict[str, Any]:
    return message.model_dump(exclude_none=True)


def _json_data_from_sse(line: str) -> str | None:
    if line.startswith("data:"):
        return line[5:].strip()
    if line and not line.startswith(":"):
        return line.strip()
    return None


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class BaseLLMAdapter:
    provider_id = "base"
    endpoint = ""
    requires_api_key = True

    def __init__(self, config: LLMProviderConfig, transport: HTTPTransport | None = None):
        self.config = config
        self.transport = transport or HTTPTransport()

    @property
    def url(self) -> str:
        return f"{self.config.base_url}{self.endpoint}"

    def complete(self, request: ChatCompletionRequest, api_key: str | None) -> ChatCompletionResponse:
        self._require_api_key(api_key)
        response = self.transport.post_json(self.url, self.headers(api_key), self.body(request))
        return self.normalize_response(request, response)

    def stream(self, request: ChatCompletionRequest, api_key: str | None) -> Iterator[str]:
        self._require_api_key(api_key)
        body = self.body(request)
        body["stream"] = True
        yield from self.normalize_stream(self.transport.stream_json(self.url, self.headers(api_key), body), request)

    def _require_api_key(self, api_key: str | None) -> None:
        if self.requires_api_key and not api_key:
            raise ValueError(f"Missing API key for provider '{self.config.id}'")

    def headers(self, api_key: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        raise NotImplementedError

    def normalize_response(self, request: ChatCompletionRequest, response: dict[str, Any]) -> ChatCompletionResponse:
        raise NotImplementedError

    def normalize_stream(self, lines: Iterable[str], request: ChatCompletionRequest) -> Iterator[str]:
        raise NotImplementedError


class OpenAIAdapter(BaseLLMAdapter):
    provider_id = "openai"
    endpoint = "/chat/completions"

    def body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": [_message_dict(message) for message in request.messages],
            "stream": request.stream,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        body.update(request.extra or {})
        return body

    def normalize_response(self, request: ChatCompletionRequest, response: dict[str, Any]) -> ChatCompletionResponse:
        choice = (response.get("choices") or [{}])[0]
        return ChatCompletionResponse(
            success=True,
            provider=self.config.id,
            model=response.get("model") or request.model or self.config.default_model,
            message=choice.get("message") or {},
            usage=response.get("usage"),
            raw=response,
        )

    def normalize_stream(self, lines: Iterable[str], request: ChatCompletionRequest) -> Iterator[str]:
        emitted_done = False
        for line in lines:
            data = _json_data_from_sse(line)
            if not data:
                continue
            if data == "[DONE]":
                emitted_done = True
                yield _sse_event({"type": "done", "provider": self.config.id})
                continue
            payload = json.loads(data)
            choice = (payload.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            event: dict[str, Any] = {"type": "chunk", "provider": self.config.id}
            if delta.get("content") is not None:
                event["content"] = delta["content"]
            if delta.get("tool_calls") is not None:
                event["tool_calls"] = delta["tool_calls"]
            if len(event) > 2:
                yield _sse_event(event)
            if choice.get("finish_reason") and not emitted_done:
                emitted_done = True
                yield _sse_event({"type": "done", "provider": self.config.id})


class AnthropicAdapter(BaseLLMAdapter):
    provider_id = "anthropic"
    endpoint = "/messages"

    def headers(self, api_key: str | None) -> dict[str, str]:
        return {
            "x-api-key": api_key or "",
            "anthropic-version": str(self.config.extra.get("anthropic_version") or "2023-06-01"),
        }

    def body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                if isinstance(message.content, str):
                    system_parts.append(message.content)
                continue
            messages.append(self._convert_message(message))

        body: dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens or int(self.config.extra.get("default_max_tokens", 1024)),
            "stream": request.stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.tools is not None:
            body["tools"] = [self._convert_tool(tool) for tool in request.tools]
        if request.tool_choice is not None:
            body["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        body.update(request.extra or {})
        return body

    def _convert_message(self, message: ChatMessage) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content or "",
                    }
                ],
            }

        if message.role == "assistant" and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tool_call in message.tool_calls:
                function = tool_call.get("function") or {}
                name = tool_call.get("name") or function.get("name") or ""
                raw_input = tool_call.get("input", function.get("arguments", {}))
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id") or "",
                        "name": name,
                        "input": self._parse_tool_input(raw_input),
                    }
                )
            return {"role": "assistant", "content": content}

        return {"role": message.role, "content": message.content or ""}

    def _convert_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        if tool.get("type") == "function":
            function = tool.get("function") or {}
            return {
                "name": function.get("name"),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        return tool

    def _convert_tool_choice(self, choice: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(choice, str):
            if choice in {"auto", "any"}:
                return {"type": choice}
            return {"type": "tool", "name": choice}
        if choice.get("type") == "function":
            function = choice.get("function") or {}
            return {"type": "tool", "name": function.get("name")}
        if choice.get("type") in {"auto", "any", "tool"}:
            return choice
        return {"type": "auto"}

    def _parse_tool_input(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"value": value}
        return {}

    def normalize_response(self, request: ChatCompletionRequest, response: dict[str, Any]) -> ChatCompletionResponse:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    }
                )
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return ChatCompletionResponse(
            success=True,
            provider=self.config.id,
            model=response.get("model") or request.model or self.config.default_model,
            message=message,
            usage=response.get("usage"),
            raw=response,
        )

    def normalize_stream(self, lines: Iterable[str], request: ChatCompletionRequest) -> Iterator[str]:
        current_event: str | None = None
        for line in lines:
            if line.startswith("event:"):
                current_event = line[6:].strip()
                continue
            data = _json_data_from_sse(line)
            if not data:
                continue
            payload = json.loads(data)
            event_name = payload.get("type") or current_event
            if event_name in {"message_stop", "done"}:
                yield _sse_event({"type": "done", "provider": self.config.id, "event": event_name})
                current_event = None
                continue
            chunk: dict[str, Any] = {"type": "chunk", "provider": self.config.id}
            if event_name:
                chunk["event"] = event_name
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta" or "text" in delta:
                chunk["content"] = delta.get("text", "")
            content_block = payload.get("content_block") or {}
            if content_block.get("type") == "tool_use":
                chunk["tool_calls"] = [
                    {
                        "id": content_block.get("id"),
                        "type": "function",
                        "function": {
                            "name": content_block.get("name"),
                            "arguments": json.dumps(content_block.get("input") or {}, ensure_ascii=False),
                        },
                    }
                ]
            if "content" in chunk or "tool_calls" in chunk:
                yield _sse_event(chunk)
            current_event = None


class OllamaAdapter(BaseLLMAdapter):
    provider_id = "ollama"
    endpoint = "/api/chat"
    requires_api_key = False

    def body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": [_message_dict(message) for message in request.messages],
            "stream": request.stream,
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            body["options"] = options
        if request.tools is not None:
            body["tools"] = request.tools
        body.update(request.extra or {})
        return body

    def headers(self, api_key: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def normalize_response(self, request: ChatCompletionRequest, response: dict[str, Any]) -> ChatCompletionResponse:
        usage = {
            "prompt_tokens": response.get("prompt_eval_count"),
            "completion_tokens": response.get("eval_count"),
        }
        return ChatCompletionResponse(
            success=True,
            provider=self.config.id,
            model=response.get("model") or request.model or self.config.default_model,
            message=response.get("message") or {"role": "assistant", "content": response.get("response", "")},
            usage={key: value for key, value in usage.items() if value is not None} or None,
            raw=response,
        )

    def normalize_stream(self, lines: Iterable[str], request: ChatCompletionRequest) -> Iterator[str]:
        for line in lines:
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("done"):
                yield _sse_event({"type": "done", "provider": self.config.id})
                continue
            message = payload.get("message") or {}
            chunk: dict[str, Any] = {"type": "chunk", "provider": self.config.id}
            if message.get("content") is not None:
                chunk["content"] = message.get("content")
            if message.get("tool_calls") is not None:
                chunk["tool_calls"] = message.get("tool_calls")
            if len(chunk) > 2:
                yield _sse_event(chunk)


ADAPTERS = {
    OpenAIAdapter.provider_id: OpenAIAdapter,
    AnthropicAdapter.provider_id: AnthropicAdapter,
    OllamaAdapter.provider_id: OllamaAdapter,
}


def adapter_for(config: LLMProviderConfig, transport: HTTPTransport | None = None) -> BaseLLMAdapter:
    adapter_cls = ADAPTERS.get(config.id)
    if adapter_cls is None:
        raise ValueError(f"Unsupported LLM provider '{config.id}'")
    return adapter_cls(config, transport=transport)
