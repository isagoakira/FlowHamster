import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import llm
from backend.schemas.llm_request import ChatCompletionRequest
from backend.schemas.llm_response import ChatCompletionResponse
from backend.services.llm_adapters import (
    AnthropicAdapter,
    EncryptedApiKeyStore,
    LLMProviderConfig,
    OllamaAdapter,
    OpenAIAdapter,
    ProviderConfigLoader,
)


class FakeTransport:
    def __init__(self, response=None, lines=None):
        self.response = response or {}
        self.lines = lines or []
        self.calls = []

    def post_json(self, url, headers, body, timeout=60):
        self.calls.append({"url": url, "headers": headers, "body": body})
        return self.response

    def stream_json(self, url, headers, body, timeout=60):
        self.calls.append({"url": url, "headers": headers, "body": body})
        yield from self.lines


def request_payload(**overrides):
    payload = {
        "provider": "openai",
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


def parse_sse(events):
    parsed = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data:"):
                parsed.append(json.loads(line[5:].strip()))
    return parsed


class LLMAdapterTest(unittest.TestCase):
    def test_config_loader_hot_reloads_provider_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "providers.json"
            config_path.write_text(
                json.dumps({"providers": [{"id": "openai", "name": "OpenAI", "base_url": "https://example", "models": ["a"]}]}),
                encoding="utf-8",
            )
            loader = ProviderConfigLoader(config_path)

            self.assertEqual([provider.id for provider in loader.providers()], ["openai"])

            time.sleep(0.01)
            config_path.write_text(
                json.dumps({"providers": [{"id": "ollama", "name": "Ollama", "base_url": "http://localhost", "models": ["b"]}]}),
                encoding="utf-8",
            )
            os.utime(config_path, None)

            self.assertEqual([provider.id for provider in loader.providers()], ["ollama"])

    def test_api_key_store_does_not_store_plaintext(self):
        store = EncryptedApiKeyStore(secret="test-secret")
        store.set("openai", "sk-test")

        self.assertEqual(store.get("openai"), "sk-test")
        self.assertNotIn("sk-test", store._values.values())

        store.delete("openai")
        self.assertIsNone(store.get("openai"))

    def test_openai_body_preserves_chat_completion_fields(self):
        adapter = OpenAIAdapter(LLMProviderConfig("openai", "OpenAI", "https://api.test/v1", ["gpt"], "gpt"))
        req = request_payload(
            provider="openai",
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            tool_choice="auto",
            temperature=0.2,
            max_tokens=32,
            extra={"seed": 7},
        )

        body = adapter.body(req)

        self.assertEqual(body["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["seed"], 7)

    def test_openai_stream_normalizes_content_and_done(self):
        adapter = OpenAIAdapter(LLMProviderConfig("openai", "OpenAI", "https://api.test/v1", ["gpt"], "gpt"))
        lines = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            "data: [DONE]",
        ]

        events = parse_sse(adapter.normalize_stream(lines, request_payload(provider="openai")))

        self.assertEqual(events[0]["content"], "he")
        self.assertEqual(events[1]["content"], "llo")
        self.assertEqual(events[-1], {"type": "done", "provider": "openai"})

    def test_anthropic_body_converts_system_tool_calls_and_tool_results(self):
        adapter = AnthropicAdapter(
            LLMProviderConfig(
                "anthropic",
                "Anthropic",
                "https://api.test/v1",
                ["claude"],
                "claude",
                extra={"default_max_tokens": 512},
            )
        )
        req = request_payload(
            provider="anthropic",
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            ],
        )

        body = adapter.body(req)

        self.assertEqual(body["system"], "Be helpful.")
        self.assertEqual(body["messages"][1]["content"][0]["type"], "tool_use")
        self.assertEqual(body["messages"][1]["content"][0]["input"], {"city": "NYC"})
        self.assertEqual(body["messages"][2]["role"], "user")
        self.assertEqual(body["messages"][2]["content"][0]["type"], "tool_result")

    def test_anthropic_tool_choice_auto_maps_to_anthropic_shape(self):
        adapter = AnthropicAdapter(LLMProviderConfig("anthropic", "Anthropic", "https://api.test/v1", ["claude"], "claude"))
        body = adapter.body(request_payload(provider="anthropic", tool_choice="auto"))

        self.assertEqual(body["tool_choice"], {"type": "auto"})

    def test_anthropic_tool_choice_function_maps_to_named_tool(self):
        adapter = AnthropicAdapter(LLMProviderConfig("anthropic", "Anthropic", "https://api.test/v1", ["claude"], "claude"))
        body = adapter.body(
            request_payload(
                provider="anthropic",
                tool_choice={"type": "function", "function": {"name": "lookup"}},
            )
        )

        self.assertEqual(body["tool_choice"], {"type": "tool", "name": "lookup"})

    def test_anthropic_stream_preserves_event_and_normalizes_done(self):
        adapter = AnthropicAdapter(LLMProviderConfig("anthropic", "Anthropic", "https://api.test/v1", ["claude"], "claude"))
        lines = [
            "event: content_block_delta",
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]

        events = parse_sse(adapter.normalize_stream(lines, request_payload(provider="anthropic")))

        self.assertEqual(events[0]["event"], "content_block_delta")
        self.assertEqual(events[0]["content"], "hi")
        self.assertNotIn("data: event:", "".join(adapter.normalize_stream(lines, request_payload(provider="anthropic"))))
        self.assertEqual(events[-1]["type"], "done")

    def test_ollama_body_maps_max_tokens_to_num_predict(self):
        adapter = OllamaAdapter(LLMProviderConfig("ollama", "Ollama", "http://localhost:11434", ["llama"], "llama"))
        body = adapter.body(request_payload(provider="ollama", max_tokens=128, temperature=0.3))

        self.assertEqual(body["options"]["num_predict"], 128)
        self.assertEqual(body["options"]["temperature"], 0.3)


class LLMRouterTest(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(llm.router, prefix="/api")
        self.client = TestClient(self.app)

    def test_provider_and_api_key_lifecycle(self):
        provider = LLMProviderConfig("openai", "OpenAI", "https://api.test/v1", ["gpt"], "gpt")
        loader = _StaticLoader(provider)
        store = EncryptedApiKeyStore(secret="test-secret")

        with patch.object(llm, "config_loader", loader), patch.object(llm, "api_key_store", store):
            response = self.client.get("/api/llm/providers")
            self.assertFalse(response.json()["providers"][0]["api_key_configured"])

            response = self.client.put("/api/llm/providers/openai/api-key", json={"api_key": "sk-test"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["api_key_configured"])

            response = self.client.delete("/api/llm/providers/openai/api-key")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["api_key_configured"])

    def test_non_stream_chat_completion_returns_normalized_response(self):
        provider = LLMProviderConfig("openai", "OpenAI", "https://api.test/v1", ["gpt"], "gpt")
        adapter = _FakeAdapter(ChatCompletionResponse(success=True, provider="openai", model="gpt", message={"role": "assistant", "content": "ok"}))

        with patch.object(llm, "config_loader", _StaticLoader(provider)), patch.object(llm, "adapter_for", lambda _: adapter):
            response = self.client.post(
                "/api/llm/chat/completions",
                json={"provider": "openai", "model": "gpt", "messages": [{"role": "user", "content": "hi"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"]["content"], "ok")

    def test_stream_chat_completion_returns_text_event_stream(self):
        provider = LLMProviderConfig("ollama", "Ollama", "http://localhost:11434", ["llama"], "llama")
        adapter = _FakeAdapter(
            stream_events=[
                'data: {"type":"chunk","provider":"ollama","content":"ok"}\n\n',
                'data: {"type":"done","provider":"ollama"}\n\n',
            ]
        )

        with patch.object(llm, "config_loader", _StaticLoader(provider)), patch.object(llm, "adapter_for", lambda _: adapter):
            response = self.client.post(
                "/api/llm/chat/completions",
                json={"provider": "ollama", "model": "llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('"type":"done"', response.text.replace(" ", ""))

    def test_scanner_style_imports(self):
        import backend.adapters
        import backend.config.provider_config
        import backend.main
        import backend.schemas.llm_request
        import backend.schemas.llm_response

        self.assertEqual(llm.CHAT_COMPLETIONS_COMPAT_KEY, "chat.completions")
        self.assertEqual(backend.main.app.title, "FlowHamster API")
        self.assertTrue(hasattr(backend.adapters, "OpenAIAdapter"))
        self.assertTrue(hasattr(backend.config.provider_config, "ProviderConfigLoader"))
        self.assertTrue(hasattr(backend.schemas.llm_request, "ChatCompletionRequest"))
        self.assertTrue(hasattr(backend.schemas.llm_response, "ChatCompletionResponse"))


class _StaticLoader:
    def __init__(self, *providers):
        self._providers = {provider.id: provider for provider in providers}

    def providers(self):
        return list(self._providers.values())

    def get(self, provider_id):
        return self._providers.get(provider_id)


class _FakeAdapter:
    def __init__(self, response=None, stream_events=None):
        self.response = response
        self.stream_events = stream_events or []

    def complete(self, request, api_key):
        return self.response

    def stream(self, request, api_key):
        yield from self.stream_events


if __name__ == "__main__":
    unittest.main()
