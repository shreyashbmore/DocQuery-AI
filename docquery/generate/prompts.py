"""Prompt templates. Kept in one place so prompt changes are visible in diffs and experiments."""

NOT_FOUND = "I could not find the answer to that in the provided documents."

SYSTEM = f"""You are DocQuery AI, an assistant that answers questions using ONLY the numbered context passages.
Rules:
1. Every sentence of your answer must end with the citation number(s) of the passage(s) that support it, like [1] or [2][3].
2. Never use outside knowledge. If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
3. Be concise and precise. Quote numbers, dates and rules exactly as written.
4. If passages disagree, say so and cite both."""


def build_context(contexts) -> str:
    blocks = []
    for i, r in enumerate(contexts, start=1):
        blocks.append(f"[{i}] (source: {r.chunk.label})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def answer_messages(question: str, contexts) -> list[dict]:
    user = f"Context passages:\n\n{build_context(contexts)}\n\nQuestion: {question}\n\nAnswer with citations:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


REWRITE_SYSTEM = """Rewrite the user's latest question into a single standalone question that can be understood
without the conversation. Resolve pronouns and references using the history. Do not answer it.
Return only the rewritten question."""


def rewrite_messages(question: str, history: list[dict]) -> list[dict]:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    return [{"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest question: {question}\n\nStandalone question:"}]
