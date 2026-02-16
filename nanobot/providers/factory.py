"""Shared provider factory for CLI, scripts, and services."""

from __future__ import annotations

from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import OpenAICodexAuthStore, OpenAICodexProvider


class ProviderInitError(RuntimeError):
    """Raised when provider configuration/auth is missing."""


def create_provider(config, model: str | None = None):
    """Create provider based on config/model.

    Selection rules:
    - model contains `codex` -> OpenAICodexProvider (OAuth via `codex login` or explicit bearer token)
    - otherwise -> LiteLLMProvider (API-key based)
    """
    selected_model = model or config.agents.defaults.model
    provider_cfg = config.get_provider(selected_model)
    provider_name = config.get_provider_name(selected_model)

    if provider_name == "openai_codex":
        auth_store = OpenAICodexAuthStore()
        if not ((provider_cfg and provider_cfg.api_key) or auth_store.has_auth()):
            raise ProviderInitError(
                "OpenAI Codex auth not configured. Run `codex login` (ChatGPT login) "
                "or set providers.openaiCodex.apiKey in ~/.nanobot/config.json."
            )
        return OpenAICodexProvider(
            api_key=provider_cfg.api_key if provider_cfg else None,
            api_base=config.get_api_base(selected_model),
            default_model=selected_model,
            auth_store=auth_store,
        )

    if not (provider_cfg and provider_cfg.api_key) and not selected_model.startswith("bedrock/"):
        if OpenAICodexAuthStore().has_auth():
            raise ProviderInitError(
                "No API-key provider configured for model "
                f"'{selected_model}'. To use Codex OAuth, set agents.defaults.model to a Codex "
                "model (e.g. gpt-5-codex) or pass --model gpt-5-codex."
            )
        raise ProviderInitError(
            "No API key configured. Set one in ~/.nanobot/config.json under providers."
        )

    return LiteLLMProvider(
        api_key=provider_cfg.api_key if provider_cfg else None,
        api_base=config.get_api_base(selected_model),
        default_model=selected_model,
        extra_headers=provider_cfg.extra_headers if provider_cfg else None,
        provider_name=provider_name,
    )
