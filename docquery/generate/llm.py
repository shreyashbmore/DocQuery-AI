"""LLM providers behind one ``chat(messages) -> str`` interface.

* ollama      - local open-source models (Llama 3.1, Qwen 2.5, Mistral ...)
* openai      - any OpenAI-compatible endpoint (OpenAI, Gemini, Groq, Together, vLLM ...)
* extractive  - no model at all: returns the best-matching cited sentences (offline baseline / tests)
"""
from __future__ import annotations

import os
import re

import httpx


def strip_thinking(text: str) -> str:
    """Reasoning models (e.g. Qwen3, DeepSeek-R1) may prefix answers with <think>...</think>."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class LLM:
    name: str

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 700, timeout: float = 120):
        self.name = f"ollama/{model}"
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages, json_mode=False):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}}
        if json_mode:
            body["format"] = "json"
        r = self.client.post(f"{self.host}/api/chat", json=body)
        r.raise_for_status()
        return strip_thinking(r.json()["message"]["content"])


class OpenAICompatibleLLM(LLM):
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 700, timeout: float = 60):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
        self.name = f"openai/{model}"
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        self.last_usage: dict = {}

    def chat(self, messages, json_mode=False):
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.client.post(f"{self.base}/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        self.last_usage = data.get("usage", {})
        return strip_thinking(data["choices"][0]["message"]["content"])


class ExtractiveLLM(LLM):
    """Picks the context sentences with the highest word overlap with the question."""

    name = "extractive"

    def chat(self, messages, json_mode=False):
        from docquery.ingest.chunkers import split_sentences
        from docquery.generate.prompts import NOT_FOUND
        from docquery.retrieve.bm25 import tokenize

        user = messages[-1]["content"]
        if "Standalone question:" in user:  # query rewriting: return question unchanged
            return re.search(r"Latest question: (.*)\n", user).group(1)
        question = re.search(r"Question: (.*)\n", user).group(1)
        q = set(tokenize(question))
        scored = []
        for num, body in re.findall(r"\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=\n\n\[\d+\] \(source|\n\nQuestion:)", user, re.S):
            for sent in split_sentences(body):
                overlap = len(q & set(tokenize(sent))) / max(1, len(q))
                scored.append((overlap, sent, num))
        scored.sort(key=lambda x: -x[0])
        best = [s for s in scored[:2] if s[0] > 0.2]
        if not best:
            return NOT_FOUND
        return " ".join(f"{s.rstrip('.')} [{n}]." for _, s, n in best)


def build_llm(cfg: dict) -> LLM:
    provider = cfg.get("provider", "ollama")
    kw = dict(temperature=float(cfg.get("temperature", 0.1)), max_tokens=int(cfg.get("max_tokens", 700)),
              timeout=float(cfg.get("timeout", 120)))
    if provider == "ollama":
        return OllamaLLM(cfg["model"], **kw)
    if provider == "openai":
        return OpenAICompatibleLLM(cfg["model"], **kw)
    if provider == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM provider: {provider}")
