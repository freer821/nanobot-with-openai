import base64
import json
import time
from typing import Any

import httpx
import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config
from nanobot.providers.factory import create_provider
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import OpenAICodexAuthStore, OpenAICodexProvider


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


def _jwt(payload: dict[str, Any]) -> str:
    header = {"alg": "none", "typ": "JWT"}
    parts = []
    for part in (header, payload):
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        parts.append(base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("="))
    parts.append("signature")
    return ".".join(parts)


def test_config_matches_openai_codex_without_api_key() -> None:
    cfg = Config.model_validate({
        "agents": {"defaults": {"model": "gpt-5-codex"}},
    })
    assert cfg.get_provider_name() == "openai_codex"


@pytest.mark.asyncio
async def test_auth_store_reads_tokens_from_auth_json(tmp_path) -> None:
    future_exp = int(time.time()) + 3600
    access_token = _jwt({"exp": future_exp, "chatgpt_account_id": "acct_123"})
    id_token = _jwt({"chatgpt_account_id": "acct_123"})
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": access_token,
                "refresh_token": "refresh_123",
                "id_token": id_token,
            },
        }),
        encoding="utf-8",
    )

    store = OpenAICodexAuthStore(auth_path=auth_path)
    token, account_id = await store.get_access_token()
    assert token == access_token
    assert account_id == "acct_123"


def test_openai_codex_provider_build_payload_and_parse_tool_call() -> None:
    provider = OpenAICodexProvider(api_key="token", default_model="gpt-5-codex")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Open README.md"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "file content"},
    ]
    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file",
            "parameters": {"type": "object"},
        },
    }]

    payload = provider._build_payload(
        messages=messages,
        tools=tools,
        model="gpt-5-codex",
        max_tokens=2048,
        temperature=0.2,
    )
    assert payload["model"] == "gpt-5-codex"
    assert payload["instructions"] == "You are helpful."
    assert payload["tools"][0]["name"] == "read_file"

    response = httpx.Response(
        200,
        json={
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Calling tool..."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "list_dir",
                    "arguments": "{\"path\":\".\"}",
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    )
    parsed = provider._parse_response(response)
    assert parsed.content == "Calling tool..."
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "list_dir"
    assert parsed.tool_calls[0].arguments == {"path": "."}


def test_openai_codex_provider_sanitizes_surrogates_for_utf8_json() -> None:
    provider = OpenAICodexProvider(api_key="token", default_model="gpt-5-codex")
    bad = "告诉我\udce9\udc95如何减肥"
    payload = provider._build_payload(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": bad},
        ],
        tools=None,
        model="gpt-5-codex",
        max_tokens=1024,
        temperature=0.7,
    )

    # This mimics httpx JSON encoding path (ensure_ascii=False + UTF-8 bytes).
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert encoded


def test_factory_selects_openai_codex_from_model_and_config() -> None:
    cfg = Config.model_validate({
        "agents": {"defaults": {"model": "gpt-5-codex"}},
        "providers": {"openaiCodex": {"apiKey": "dummy-token"}},
    })
    provider = create_provider(cfg, model=cfg.agents.defaults.model)
    assert isinstance(provider, OpenAICodexProvider)


def test_factory_selects_litellm_for_api_key_provider() -> None:
    cfg = Config.model_validate({
        "agents": {"defaults": {"model": "gpt-4o"}},
        "providers": {"openai": {"api_key": "sk-test"}},
    })
    provider = create_provider(cfg, model=cfg.agents.defaults.model)
    assert isinstance(provider, LiteLLMProvider)
