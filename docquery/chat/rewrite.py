"""Conversational query rewriting so follow-up questions retrieve the right passages."""
from __future__ import annotations

from docquery.generate.prompts import rewrite_messages


def standalone_question(question: str, history: list[dict], llm, max_turns: int = 4) -> str:
    if not history:
        return question
    recent = history[-2 * max_turns:]
    try:
        out = llm.chat(rewrite_messages(question, recent)).strip().strip('"')
        return out.splitlines()[0] if out else question
    except Exception:
        return question
