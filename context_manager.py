"""
Conversation + tool-result context management (Phase 12).

Keeps a fixed system prompt plus a rolling list of (role, content) messages,
and truncates from the *oldest* non-system message once a rough token
estimate exceeds the configured budget -- rather than ever dumping an
entire repository or unlimited history into a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 chars/token for English prose/code). Good enough
    for a truncation heuristic; not meant to match the model's real
    tokenizer exactly."""
    return max(1, len(text) // 4)


@dataclass
class ContextManager:
    system_prompt: str
    max_context_tokens: int = 6000
    reserve_for_reply: int = 1000
    messages: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_result(self, tool_name: str, result_text: str) -> None:
        self.messages.append(
            {"role": "tool", "name": tool_name, "content": result_text}
        )

    def reset(self) -> None:
        self.messages = []

    def _budget(self) -> int:
        return self.max_context_tokens - self.reserve_for_reply - estimate_tokens(
            self.system_prompt
        )

    def truncate(self) -> None:
        """Drop oldest messages until the running estimate fits the budget.
        Always leaves at least the single most recent message, even if it
        alone exceeds budget (better to send something than nothing)."""
        budget = self._budget()
        total = sum(estimate_tokens(m["content"]) for m in self.messages)
        while total > budget and len(self.messages) > 1:
            dropped = self.messages.pop(0)
            total -= estimate_tokens(dropped["content"])

    def build_messages(self) -> list[dict]:
        """Return a chat-API-ready message list: system prompt + truncated
        history. Tool-result messages are folded into 'user' role (most
        local chat APIs don't have a first-class 'tool' role for arbitrary
        models), clearly tagged so the model can tell them apart from a
        real user message."""
        self.truncate()
        out = [{"role": "system", "content": self.system_prompt}]
        for m in self.messages:
            if m["role"] == "tool":
                out.append(
                    {"role": "user", "content": f"[tool result: {m['name']}]\n{m['content']}"}
                )
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out
