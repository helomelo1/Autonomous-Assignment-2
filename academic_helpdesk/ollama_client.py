"""Minimal Ollama HTTP client; no LangChain or LiteLLM dependency required."""
from __future__ import annotations

import requests
from typing import Any

from . import config


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = config.OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.1) -> str:
        payload = {
            "model": model or config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            # Qwen3 supports Ollama's explicit thinking toggle. Keep it off
            # for this retrieval/tool-oriented helpdesk to reduce latency.
            "think": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            content = body.get("message", {}).get("content", "")
            if not content:
                raise OllamaError("Ollama returned an empty chat response.")
            return str(content).strip()
        except requests.RequestException as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and pull {config.LLM_MODEL}. Details: {exc}"
            ) from exc

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        model = model or config.EMBEDDING_MODEL
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=config.REQUEST_TIMEOUT,
            )
            if response.ok:
                embeddings = response.json().get("embeddings")
                if embeddings and len(embeddings) == len(texts):
                    return embeddings
            # Compatibility fallback for Ollama versions exposing the older endpoint.
            output: list[list[float]] = []
            for text in texts:
                old = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=config.REQUEST_TIMEOUT,
                )
                old.raise_for_status()
                embedding = old.json().get("embedding")
                if not embedding:
                    raise OllamaError("Ollama returned no embedding vector.")
                output.append(embedding)
            return output
        except requests.RequestException as exc:
            raise OllamaError(
                f"Could not create embeddings with Ollama model '{model}'. "
                f"Run `ollama pull {model}` and verify Ollama is running. Details: {exc}"
            ) from exc
