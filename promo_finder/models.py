"""The only place that talks to a generative model.

Swap the model by changing `[finder]` in config.toml (or the Streamlit sidebar).
Two wire formats cover almost everything:
  - "openai": any OpenAI-compatible /chat/completions endpoint
              (Gemini, Groq, OpenRouter, Ollama, vLLM, OpenAI, a Kuku gateway)
  - "anthropic": Claude's Messages API
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

PRESETS = {
    "gemini":     {"api": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                   "model": "gemini-3.8-flash", "key_env": "GEMINI_API_KEY"},
    "groq":       {"api": "openai", "base_url": "https://api.groq.com/openai/v1",
                   "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY"},
    "openrouter": {"api": "openai", "base_url": "https://openrouter.ai/api/v1",
                   "model": "google/gemini-3.5-flash", "key_env": "OPENROUTER_API_KEY"},
    "deepseek":   {"api": "openai", "base_url": "https://api.deepseek.com",
                   "model": "deepseek-flash", "key_env": "DEEPSEEK_API_KEY"},
    "ollama":     {"api": "openai", "base_url": "http://localhost:11434/v1",
                   "model": "qwen2.5:7b", "key_env": ""},
    "anthropic":  {"api": "anthropic", "base_url": "https://api.anthropic.com/v1",
                   "model": "claude-haiku-5-5", "key_env": "ANTHROPIC_API_KEY"},
}

CACHE = Path(".cache/llm")


class ModelError(RuntimeError):
    pass


def resolve(cfg: dict) -> dict:
    """Merge a config block over its preset. Any field can be overridden."""
    out = dict(PRESETS.get(cfg.get("provider", ""), {}))
    out.update({k: v for k, v in cfg.items() if v not in (None, "")})
    if not out.get("api_key") and out.get("key_env"):
        out["api_key"] = os.environ.get(out["key_env"], "")
    return out


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ModelError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _post(url, headers, body, timeout=180):
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code in (429, 500, 502, 503, 504):
        raise ModelError(f"retryable {r.status_code}: {r.text[:300]}")
    if r.status_code >= 400:
        raise ModelError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def _call_once(cfg: dict, system: str, prompt: str, schema: dict | None) -> tuple[str, dict]:
    if cfg["api"] == "anthropic":
        body = {"model": cfg["model"], "max_tokens": cfg.get("max_tokens", 8000),
                "temperature": cfg.get("temperature", 0.3), "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        data = _post(f"{cfg['base_url']}/messages",
                     {"x-api-key": cfg["api_key"], "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in data.get("content", []))
        u = data.get("usage", {})
        return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}

    body = {"model": cfg["model"], "temperature": cfg.get("temperature", 0.3),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if schema and cfg.get("structured", True):
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "result", "schema": schema}}
    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
    try:
        data = _post(f"{cfg['base_url']}/chat/completions", headers, body)
    except ModelError as e:
        # Some OpenAI-compatible servers reject json_schema; fall back to plain text + parse.
        if "response_format" in body and str(e).startswith("HTTP 400"):
            body.pop("response_format")
            data = _post(f"{cfg['base_url']}/chat/completions", headers, body)
        else:
            raise
    text = data["choices"][0]["message"].get("content") or ""
    u = data.get("usage") or {}
    return text, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}


def call_model(prompt: str, schema: dict | None, cfg: dict, system: str = "",
               retries: int = 4, use_cache: bool = True) -> dict:
    """Send one prompt, get back {"json": parsed, "raw": text, "usage": ..., "model": ...}."""
    cfg = resolve(cfg)
    key = hashlib.sha256(json.dumps([cfg["model"], cfg["base_url"], system, prompt],
                                    sort_keys=True).encode()).hexdigest()[:24]
    hit = CACHE / f"{key}.json"
    if use_cache and hit.exists():
        return json.loads(hit.read_text()) | {"cached": True}

    last = None
    for attempt in range(retries):
        try:
            text, usage = _call_once(cfg, system, prompt, schema)
            out = {"json": extract_json(text), "raw": text, "usage": usage, "model": cfg["model"]}
            CACHE.mkdir(parents=True, exist_ok=True)
            hit.write_text(json.dumps(out))
            return out
        except (ModelError, json.JSONDecodeError, requests.RequestException) as e:
            last = e
            if str(e).startswith("HTTP 4"):  # bad key / bad model name: retrying won't help
                break
            time.sleep(min(60, 4 * 2 ** attempt))
    raise ModelError(f"{cfg['model']} failed after {retries} tries: {last}")
