"""OpenAI Codex provider backed by ChatGPT OAuth tokens."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_RESPONSES_PATH = "/responses"
DEFAULT_AUTH_PATH = Path.home() / ".codex" / "auth.json"

OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
OAUTH_CLIENT_ID = "oai-prod-uaz6nJY2N6m2NQVkq7gM6aKv"
OAUTH_SCOPE = "openid profile email offline_access"
OAUTH_REDIRECT_URI = "https://chat.openai.com/auth/callback"


class OpenAICodexAuthError(RuntimeError):
    """Raised when ChatGPT OAuth credentials are missing or invalid."""


@dataclass
class CodexTokenBundle:
    """Token bundle loaded from Codex auth.json."""

    access_token: str
    refresh_token: str | None = None
    id_token: str | None = None
    account_id: str | None = None


class OpenAICodexAuthStore:
    """Read and refresh ChatGPT OAuth tokens stored by the Codex CLI."""

    def __init__(self, auth_path: Path | None = None):
        self.auth_path = (auth_path or DEFAULT_AUTH_PATH).expanduser()

    def has_auth(self) -> bool:
        """Return True if an access token is available in auth.json."""
        try:
            return bool(self._load_tokens())
        except OpenAICodexAuthError:
            return False

    async def get_access_token(self, force_refresh: bool = False) -> tuple[str, str | None]:
        """Load a usable access token and associated ChatGPT account id."""
        bundle = self._load_tokens()

        if force_refresh or self._token_expires_soon(bundle.access_token):
            if not bundle.refresh_token:
                raise OpenAICodexAuthError(
                    "ChatGPT OAuth refresh token missing. Run `codex login` again."
                )
            bundle = await self._refresh_tokens(bundle.refresh_token, bundle)
            self._save_tokens(bundle)

        return bundle.access_token, bundle.account_id

    def _load_tokens(self) -> CodexTokenBundle:
        data = self._read_auth_json()
        if data.get("auth_mode") != "chatgpt":
            raise OpenAICodexAuthError(
                "Codex auth mode is not ChatGPT OAuth. Run `codex login` and pick ChatGPT login."
            )

        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            raise OpenAICodexAuthError(
                "Missing ChatGPT OAuth tokens in ~/.codex/auth.json. Run `codex login`."
            )

        access_token = tokens.get("access_token")
        if not access_token:
            raise OpenAICodexAuthError("Missing access_token in ~/.codex/auth.json.")

        account_id = tokens.get("account_id")
        id_token = tokens.get("id_token")
        if not account_id and id_token:
            claims = self._decode_jwt_claims(id_token)
            account_id = claims.get("chatgpt_account_id") if isinstance(claims, dict) else None

        return CodexTokenBundle(
            access_token=access_token,
            refresh_token=tokens.get("refresh_token"),
            id_token=id_token,
            account_id=account_id,
        )

    async def _refresh_tokens(
        self,
        refresh_token: str,
        previous: CodexTokenBundle | None = None,
    ) -> CodexTokenBundle:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OAUTH_CLIENT_ID,
            "scope": OAUTH_SCOPE,
            "redirect_uri": OAUTH_REDIRECT_URI,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(OAUTH_TOKEN_URL, data=payload)

        if response.status_code >= 400:
            detail = response.text.strip()[:500]
            raise OpenAICodexAuthError(
                f"Failed to refresh ChatGPT OAuth token (HTTP {response.status_code}): {detail}"
            )

        body = response.json()
        access_token = body.get("access_token")
        if not access_token:
            raise OpenAICodexAuthError("OAuth refresh succeeded but access_token is missing.")

        refreshed = CodexTokenBundle(
            access_token=access_token,
            refresh_token=body.get("refresh_token") or refresh_token,
            id_token=body.get("id_token") or (previous.id_token if previous else None),
            account_id=body.get("account_id") or (previous.account_id if previous else None),
        )

        if not refreshed.account_id and refreshed.id_token:
            claims = self._decode_jwt_claims(refreshed.id_token)
            if isinstance(claims, dict):
                refreshed.account_id = claims.get("chatgpt_account_id")

        return refreshed

    def _save_tokens(self, bundle: CodexTokenBundle) -> None:
        data = self._read_auth_json()
        tokens = data.setdefault("tokens", {})
        tokens["access_token"] = bundle.access_token
        if bundle.refresh_token:
            tokens["refresh_token"] = bundle.refresh_token
        if bundle.id_token:
            tokens["id_token"] = bundle.id_token
        if bundle.account_id:
            tokens["account_id"] = bundle.account_id

        data["auth_mode"] = "chatgpt"
        data["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(self.auth_path, 0o600)
        except OSError:
            pass

    def _read_auth_json(self) -> dict[str, Any]:
        if not self.auth_path.exists():
            raise OpenAICodexAuthError(
                f"Codex auth file not found: {self.auth_path}. Run `codex login` first."
            )
        try:
            return json.loads(self.auth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenAICodexAuthError(
                f"Invalid JSON in Codex auth file: {self.auth_path}"
            ) from exc

    def _decode_jwt_claims(self, token: str) -> dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = parts[1]
        pad_len = (4 - (len(payload) % 4)) % 4
        payload += "=" * pad_len
        try:
            raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
            claims = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        return claims if isinstance(claims, dict) else {}

    def _token_expires_soon(self, token: str) -> bool:
        claims = self._decode_jwt_claims(token)
        exp = claims.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return exp <= (time.time() + 60)


class OpenAICodexProvider(LLMProvider):
    """Provider for ChatGPT-plan-backed OpenAI Codex models."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-5-codex",
        auth_store: OpenAICodexAuthStore | None = None,
        timeout: float = 90.0,
    ):
        super().__init__(api_key=api_key, api_base=api_base or DEFAULT_CODEX_API_BASE)
        self.default_model = default_model
        self.timeout = timeout
        self.auth_store = auth_store or OpenAICodexAuthStore()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        model_name = model or self.default_model
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            token, account_id = await self._get_auth(force_refresh=False)
            response = await self._send_request(payload, token, account_id)
            if response.status_code == 401 and not self.api_key:
                token, account_id = await self._get_auth(force_refresh=True)
                response = await self._send_request(payload, token, account_id)
            return self._parse_response(response)
        except OpenAICodexAuthError as e:
            return LLMResponse(content=f"OpenAI Codex auth error: {e}", finish_reason="error")
        except Exception as e:
            return LLMResponse(content=f"Error calling OpenAI Codex: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model

    async def _get_auth(self, force_refresh: bool) -> tuple[str, str | None]:
        if self.api_key:
            return self.api_key, None
        return await self.auth_store.get_access_token(force_refresh=force_refresh)

    async def _send_request(
        self,
        payload: dict[str, Any],
        token: str,
        account_id: str | None,
    ) -> httpx.Response:
        base = (self.api_base or DEFAULT_CODEX_API_BASE).rstrip("/")
        url = f"{base}{DEFAULT_CODEX_RESPONSES_PATH}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(url, headers=headers, json=payload)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        instructions = []
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                text = self._content_to_text(msg.get("content"))
                if text:
                    instructions.append(text)
                continue
            input_items.extend(self._convert_message(msg))

        payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
            "store": False,
        }

        payload["instructions"] = "\n\n".join(instructions) if instructions else "You are a helpful AI assistant."

        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]
            payload["tool_choice"] = "auto"

        return self._sanitize_json(payload)

    def _convert_tool(self, tool_def: dict[str, Any]) -> dict[str, Any]:
        if tool_def.get("type") == "function":
            fn = tool_def.get("function", {})
            return {
                "type": "function",
                "name": self._sanitize_text(str(fn.get("name") or "")),
                "description": self._sanitize_text(str(fn.get("description") or "")),
                "parameters": fn.get("parameters", {"type": "object"}),
            }
        return tool_def

    def _convert_message(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        role = msg.get("role")
        if role == "tool":
            return [{
                "type": "function_call_output",
                "call_id": self._sanitize_text(str(msg.get("tool_call_id") or msg.get("id") or "")),
                "output": self._content_to_text(msg.get("content")),
            }]

        if role == "assistant":
            items = []
            content = msg.get("content")
            if content:
                items.append({"role": "assistant", "content": self._normalize_content(content)})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                items.append({
                    "type": "function_call",
                    "call_id": self._sanitize_text(str(tc.get("id") or "")),
                    "name": self._sanitize_text(str(fn.get("name") or "")),
                    "arguments": self._sanitize_text(str(fn.get("arguments") or "{}")),
                })
            return items

        content = msg.get("content")
        if role in {"user", "assistant"}:
            return [{"role": role, "content": self._normalize_content(content)}]
        return []

    def _normalize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._sanitize_text(content)
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({
                        "type": "input_text",
                        "text": self._sanitize_text(str(part.get("text") or "")),
                    })
                elif ptype == "image_url":
                    image = part.get("image_url", {})
                    url = image.get("url")
                    if url:
                        parts.append({
                            "type": "input_image",
                            "image_url": self._sanitize_text(str(url)),
                        })
            if parts:
                return parts
        return self._content_to_text(content)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return self._sanitize_text(content)
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(self._sanitize_text(str(part.get("text") or "")))
            return "\n".join(t for t in texts if t)
        return self._sanitize_text(str(content or ""))

    def _sanitize_text(self, value: str) -> str:
        """Normalize text to UTF-8-safe form (handles surrogate code points)."""
        if not value:
            return ""
        try:
            value.encode("utf-8")
            return value
        except UnicodeEncodeError:
            pass

        # Common case: terminal mis-decoding via surrogateescape.
        try:
            return value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        except UnicodeEncodeError:
            # Fallback: replace invalid code points directly.
            return value.encode("utf-8", "replace").decode("utf-8")

    def _sanitize_json(self, value: Any) -> Any:
        """Recursively sanitize strings before JSON serialization."""
        if isinstance(value, str):
            return self._sanitize_text(value)
        if isinstance(value, list):
            return [self._sanitize_json(v) for v in value]
        if isinstance(value, dict):
            return {k: self._sanitize_json(v) for k, v in value.items()}
        return value

    def _parse_response(self, response: httpx.Response) -> LLMResponse:
        if response.status_code >= 400:
            detail = response.text.strip()
            if len(detail) > 1000:
                detail = detail[:1000] + "... (truncated)"
            return LLMResponse(
                content=f"OpenAI Codex request failed (HTTP {response.status_code}): {detail}",
                finish_reason="error",
            )

        content_type = response.headers.get("content-type", "").lower()
        raw = response.text or ""
        data: dict[str, Any] | None = None

        # Prefer JSON when clearly marked as JSON and body is not SSE-like.
        if "application/json" in content_type and not raw.lstrip().startswith(("data:", "event:")):
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                data = None

        if data is None:
            events = self._parse_sse_events(raw)
            if events:
                data = self._sse_events_to_response(events)
            else:
                try:
                    parsed = response.json()
                    if isinstance(parsed, dict):
                        data = parsed
                except Exception as exc:
                    snippet = raw[:500] if raw else "(empty body)"
                    return LLMResponse(
                        content=f"OpenAI Codex parse error: {exc}. Raw body: {snippet}",
                        finish_reason="error",
                    )

        if data is None:
            return LLMResponse(
                content="OpenAI Codex parse error: empty/unsupported response payload.",
                finish_reason="error",
            )

        return self._parse_response_payload(data)

    def _parse_response_payload(self, data: dict[str, Any]) -> LLMResponse:
        output = data.get("output", [])
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for item in output:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")

            if itype == "message":
                for part in item.get("content", []) or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"output_text", "text"}:
                        text = part.get("text")
                        if isinstance(text, str):
                            text_parts.append(self._sanitize_text(text))
                continue

            if itype == "function_call":
                name = item.get("name") or ""
                call_id = item.get("call_id") or item.get("id") or f"call_{len(tool_calls) + 1}"
                arguments = item.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                if not isinstance(arguments, dict):
                    arguments = {"raw": str(arguments)}
                tool_calls.append(ToolCallRequest(id=call_id, name=name, arguments=arguments))
                continue

        if not text_parts:
            output_text = data.get("output_text")
            if isinstance(output_text, str):
                text_parts.append(self._sanitize_text(output_text))

        usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = usage_raw.get("input_tokens", usage_raw.get("prompt_tokens", 0)) or 0
        completion_tokens = usage_raw.get(
            "output_tokens", usage_raw.get("completion_tokens", 0)
        ) or 0
        total_tokens = usage_raw.get("total_tokens", prompt_tokens + completion_tokens) or 0
        usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }

        finish_reason = "tool_calls" if tool_calls else "stop"
        content = "\n".join(text_parts).strip() or None
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _parse_sse_events(self, raw: str) -> list[dict[str, Any]]:
        """Parse SSE payload into JSON event objects."""
        events: list[dict[str, Any]] = []
        data_lines: list[str] = []

        for line in raw.splitlines():
            if line.startswith("event:"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
                continue
            if line.strip() == "":
                if not data_lines:
                    continue
                data_str = "\n".join(data_lines).strip()
                data_lines = []
                if data_str == "[DONE]":
                    continue
                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(evt, dict):
                    events.append(evt)
                continue

        if data_lines:
            data_str = "\n".join(data_lines).strip()
            if data_str and data_str != "[DONE]":
                try:
                    evt = json.loads(data_str)
                    if isinstance(evt, dict):
                        events.append(evt)
                except json.JSONDecodeError:
                    pass

        return events

    def _sse_events_to_response(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert SSE events to a response payload shape compatible with parser."""
        final: dict[str, Any] | None = None
        text_chunks: list[str] = []
        output_items: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}

        for event in events:
            etype = event.get("type")
            if etype == "response.completed":
                resp = event.get("response")
                if isinstance(resp, dict):
                    final = resp
                continue
            if etype in {"response.output_text.delta", "output_text.delta"}:
                delta = event.get("delta")
                if isinstance(delta, str):
                    text_chunks.append(delta)
                continue
            if etype in {"response.output_item.done", "response.output_item.added"}:
                item = event.get("item")
                if isinstance(item, dict):
                    output_items.append(item)
                continue
            if etype == "response.usage" and isinstance(event.get("usage"), dict):
                usage = event["usage"]

        if final is not None:
            return final

        output: list[dict[str, Any]] = []
        if text_chunks:
            output.append({
                "type": "message",
                "content": [{"type": "output_text", "text": "".join(text_chunks)}],
            })
        output.extend(output_items)
        payload: dict[str, Any] = {"output": output}
        if usage:
            payload["usage"] = usage
        return payload
