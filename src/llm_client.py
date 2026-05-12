"""Minimal LLM client wrapper with on-disk caching and retry.

Uses OpenAI Chat Completions API. Responses are cached by hash of
(model, messages, temperature, max_tokens) so re-runs are free and
experiment-runs are resumable.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI, RateLimitError

WORKSPACE = Path(__file__).resolve().parent.parent
CACHE_DIR = WORKSPACE / "cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CLIENT: OpenAI | None = None


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


def _cache_key(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_cache: bool = True,
    retries: int = 5,
) -> dict[str, Any]:
    """Return a dict with `content`, `prompt_tokens`, `completion_tokens`, `total_tokens`."""
    key = _cache_key(model, messages, temperature, max_tokens)
    cp = _cache_path(key)
    if use_cache and cp.exists():
        try:
            return json.loads(cp.read_text())
        except json.JSONDecodeError:
            pass

    client = get_client()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            out = {
                "content": content,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "model": model,
            }
            cp.write_text(json.dumps(out))
            return out
        except (RateLimitError, APIError) as e:  # transient
            last_err = e
            wait = min(2 ** attempt + 1, 30)
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = min(2 ** attempt + 1, 30)
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


if __name__ == "__main__":
    msg = [
        {"role": "system", "content": "Reply in one sentence."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    out = chat("gpt-4.1-mini", msg, max_tokens=64)
    print(out)
