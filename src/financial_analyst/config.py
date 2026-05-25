"""LLM + embedding provider configuration.

Selects between OpenAI, GitHub Models, and Azure OpenAI based on the
``LLM_PROVIDER`` environment variable. Embedding model is fixed to a pinned
revision of ``BAAI/bge-small-en-v1.5`` for reproducibility.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from .paths import ROOT_DIR

logger = logging.getLogger(__name__)

ProviderName = Literal["openai", "github_models", "azure_openai"]

# Pinned model identifiers. Bumping these is an intentional act because it
# changes embeddings and therefore retrieval scores.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Pin the HF revision so re-syncs are deterministic.
EMBED_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"


@dataclass
class ProviderConfig:
    """Resolved provider configuration."""

    provider: ProviderName
    model: str
    extra: dict


def _load_env() -> None:
    """Load ``.env`` from the project root if present (idempotent)."""
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def get_provider_config() -> ProviderConfig:
    """Resolve the LLM provider config from env vars (no network calls)."""
    _load_env()
    provider = os.environ.get("LLM_PROVIDER", "github_models").strip().lower()
    if provider not in {"openai", "github_models", "azure_openai"}:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={provider!r}. "
            "Must be one of: openai, github_models, azure_openai."
        )

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is unset. "
                "Set it in .env."
            )
        return ProviderConfig(
            provider="openai",
            model=model,
            extra={"api_key": api_key, "api_base": base_url},
        )

    if provider == "github_models":
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        base_url = os.environ.get(
            "GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"
        ).strip()
        model = os.environ.get("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini").strip()
        if not token:
            raise RuntimeError(
                "LLM_PROVIDER=github_models but GITHUB_TOKEN is unset. "
                "Create a token with the `models:read` scope and put it in .env."
            )
        return ProviderConfig(
            provider="github_models",
            model=model,
            extra={"api_key": token, "api_base": base_url},
        )

    # azure_openai
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
    missing = [
        k for k, v in {
            "AZURE_OPENAI_API_KEY": api_key,
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
        }.items() if not v
    ]
    if missing:
        raise RuntimeError(
            f"LLM_PROVIDER=azure_openai but missing env vars: {', '.join(missing)}."
        )
    return ProviderConfig(
        provider="azure_openai",
        model=deployment,
        extra={
            "api_key": api_key,
            "azure_endpoint": endpoint,
            "azure_deployment": deployment,
            "api_version": api_version,
        },
    )


def build_llm(temperature: float = 0.1, max_tokens: int | None = 1024) -> LLM:
    """Construct a LlamaIndex LLM based on the configured provider.

    Both ``openai`` and ``github_models`` route through ``OpenAILike`` so
    that we don't hit the hardcoded-model-name validator inside the
    ``OpenAI`` class — which lags behind real-world model launches and
    refuses unknown names (including custom/internal deployments). Both
    endpoints speak the same OpenAI chat-completions REST surface, so
    ``OpenAILike`` is the right primitive.

    Newer reasoning/preview models (o-series, gpt-5*, etc.) require
    ``max_completion_tokens`` instead of ``max_tokens``. Toggle this with
    the env var ``OPENAI_USE_COMPLETION_TOKENS=true``.
    """
    cfg = get_provider_config()
    use_completion_tokens = (
        os.environ.get("OPENAI_USE_COMPLETION_TOKENS", "").strip().lower()
        in {"1", "true", "yes"}
    )

    if cfg.provider in {"openai", "github_models"}:
        kwargs: dict = {}
        if max_tokens is not None:
            if use_completion_tokens:
                # Newer/reasoning models reject `max_tokens` — use the
                # replacement param via additional_kwargs so the OpenAILike
                # base class doesn't also append the legacy `max_tokens`.
                kwargs["additional_kwargs"] = {"max_completion_tokens": max_tokens}
            else:
                kwargs["max_tokens"] = max_tokens
        return OpenAILike(
            model=cfg.model,
            api_key=cfg.extra["api_key"],
            api_base=cfg.extra["api_base"],
            temperature=temperature,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=128_000,
            **kwargs,
        )

    # Azure OpenAI uses a different class to handle the deployment-based URL.
    from llama_index.llms.openai import AzureOpenAI  # local import to keep top light

    return AzureOpenAI(
        model=cfg.model,
        deployment_name=cfg.extra["azure_deployment"],
        api_key=cfg.extra["api_key"],
        azure_endpoint=cfg.extra["azure_endpoint"],
        api_version=cfg.extra["api_version"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_embedding_model() -> HuggingFaceEmbedding:
    """Construct the pinned local embedding model."""
    return HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        revision=EMBED_MODEL_REVISION,
        embed_batch_size=32,
    )


def configure_settings(llm: LLM | None = None) -> None:
    """Wire LLM + embedding into LlamaIndex global Settings.

    Pass ``llm=None`` to defer LLM construction (useful when only retrieval
    is needed — e.g. running embeddings-only ingestion or evaluation without
    a real API key configured).
    """
    if llm is not None:
        Settings.llm = llm
    Settings.embed_model = build_embedding_model()
