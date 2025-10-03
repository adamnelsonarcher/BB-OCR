"""
Pluggable LLM providers for the BB-OCR pipeline.

Backends:
- ollama (local): POST http://127.0.0.1:11434/api/generate with base64 images
- openai: POST https://api.openai.com/v1/chat/completions with text+image parts
- gemini: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent

This module is optional and meant for testing and comparisons. The default
pipeline continues to use Ollama locally.
"""

from typing import List, Optional
import os
import base64
import requests


class LLMClient:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    def generate(self, model: str, prompt: str, images_b64: List[str], *, timeout_seconds: float = 60.0) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://127.0.0.1:11434", session: Optional[requests.Session] = None):
        super().__init__(session=session)
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, images_b64: List[str], *, timeout_seconds: float = 60.0) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "images": images_b64,
        }
        # Separate connect/read timeouts like the extractor
        connect_timeout = 2.5
        read_timeout = max(3.0, timeout_seconds - connect_timeout)
        resp = self.session.post(url, json=payload, timeout=(connect_timeout, read_timeout))
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        super().__init__(session=session)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def generate(self, model: str, prompt: str, images_b64: List[str], *, timeout_seconds: float = 60.0) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        # Build multi-part message with text + image parts using data URLs
        content: List[dict] = [{"type": "text", "text": prompt}]
        for img_b64 in images_b64:
            data_url = f"data:image/jpeg;base64,{img_b64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": 0,
        }
        resp = self.session.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""


class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        super().__init__(session=session)
        # Load .env if available (no-op if missing)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        self.base_url = os.getenv("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com")
        # Debug fields for last attempt
        self.last_url: Optional[str] = None
        self.last_model: Optional[str] = None
        self.tried_models: list[str] = []

    def generate(self, model: str, prompt: str, images_b64: List[str], *, timeout_seconds: float = 60.0) -> str:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")
        # v1beta generateContent
        target_model = (model or "gemini-1.5-flash").strip()
        url = f"{self.base_url}/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        self.tried_models = [target_model]
        self.last_url = url
        self.last_model = target_model
        parts: List[dict] = [{"text": prompt}]
        for img_b64 in images_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64,
                }
            })
        payload = {
            "contents": [{"parts": parts}],
        }
        def _post(u: str, m: str) -> requests.Response:
            self.last_url = u
            self.last_model = m
            if m not in self.tried_models:
                self.tried_models.append(m)
            return self.session.post(u, json=payload, timeout=timeout_seconds)

        resp = _post(url, target_model)
        # If model path returns 404, try constrained fallbacks within supported set
        if resp.status_code == 404:
            preferred = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro", "gemini-2.0-flash"]
            for m in preferred:
                if m == target_model:
                    continue
                u = f"{self.base_url}/v1beta/models/{m}:generateContent?key={self.api_key}"
                r = _post(u, m)
                if r.status_code != 404:
                    resp = r
                    break
        # Handle quota/policy gracefully: 403/429 → attempt softer fallback
        if resp.status_code in (403, 429):
            # Try allowed lighter/cheaper variants first (restricted to supported set)
            try_models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]
            for m in try_models:
                u = f"{self.base_url}/v1beta/models/{m}:generateContent?key={self.api_key}"
                r = _post(u, m)
                if r.status_code < 400:
                    resp = r
                    break
        resp.raise_for_status()
        data = resp.json()
        try:
            # Concatenate candidate text parts if present
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            t = candidates[0].get("content", {}).get("parts", [])
            return "".join([p.get("text", "") for p in t])
        except Exception:
            return ""


def create_llm_client(backend: str, *, session: Optional[requests.Session] = None) -> LLMClient:
    b = (backend or "").strip().lower()
    if b in ("openai", "gpt", "gpt-4", "gpt-4o", "gpt-4o-mini"):
        return OpenAIClient(session=session)
    if b in ("gemini", "google", "gemini-1.5", "gemini-1.5-flash", "gemini-1.5-pro"):
        return GeminiClient(session=session)
    # default to ollama
    return OllamaClient(session=session)


