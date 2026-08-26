"""
Parses model output for a tool call, applies the safety policy, and
dispatches to agent_tools (Phase 8/9 glue).

The model must never be allowed to claim a tool ran when it did not: any
tool call that gets dispatched here always returns the tool's own real
stdout/stderr/exit_code (or an explicit error/denial string) back to the
model as the next message -- there is no path that fabricates a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import agent_tools
from safety_policy import Safety, classify_tool


@dataclass
class ParsedToolCall:
    name: str
    args: dict


@dataclass
class MalformedToolCall:
    """The model clearly attempted a tool call (named a tool) but the
    shape was wrong -- e.g. `args` was a bare string instead of a JSON
    object. Distinguished from `None` (not a tool call at all) so the
    caller can feed a corrective error back to the model and let it retry,
    instead of silently showing the raw malformed JSON to the user as if
    it were a plain reply."""

    name: str
    error: str


def parse_tool_call(text: str) -> ParsedToolCall | MalformedToolCall | None:
    """Return a ParsedToolCall if `text` is (only) a well-formed JSON
    tool-call object, a MalformedToolCall if it's clearly an attempted
    tool call with the wrong shape, or None if it should be treated as a
    plain reply to the user.

    Accepts either {"tool": name, "args": {...}} (the convention this
    project's system prompt asks the model to use) or
    {"name": name, "arguments": {...}} (the shape observed from Ollama's
    own tool-calling template for some models), so either convention works
    regardless of which one a given model actually follows.
    """
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    for name_key, args_key in (("tool", "args"), ("name", "arguments")):
        if name_key not in obj:
            continue
        name = obj[name_key]
        if not isinstance(name, str):
            continue
        if args_key not in obj:
            # Many tools (list_files, run_pytest, git_status, git_diff, ...)
            # take only optional arguments -- an omitted args key is a
            # valid "call with defaults", not a malformed call.
            return ParsedToolCall(name, {})
        args = obj[args_key]
        if isinstance(args, dict):
            return ParsedToolCall(name, args)
        return MalformedToolCall(
            name,
            f"{args_key!r} must be a JSON object of named arguments, "
            f"e.g. {{\"{args_key}\": {{\"path\": ...}}}} -- got "
            f"{type(args).__name__}: {args!r}",
        )
    return None


# (tool_name, args) -> approved?
ApprovalHook = Callable[[str, dict], bool]


def execute_tool_call(
    project_root: Path,
    call: ParsedToolCall,
    approval_hook: ApprovalHook,
) -> str:
    """Run the requested tool if it's known and (safe or approved).
    Always returns a real, human-readable result string -- for unknown
    tools, bad arguments, and denied approvals too -- so the caller never
    has to guess whether something actually happened."""
    fn = agent_tools.TOOLS.get(call.name)
    if fn is None:
        return f"[tool error] unknown tool: {call.name!r}"

    if classify_tool(call.name) is Safety.REQUIRES_APPROVAL:
        if not approval_hook(call.name, call.args):
            return f"[tool denied] user did not approve {call.name}({call.args})"

    try:
        result = fn(project_root, **call.args)
    except TypeError as e:
        return f"[tool error] bad arguments for {call.name}: {e}"

    status = "OK" if result.ok else f"FAILED (exit {result.exit_code})"
    parts = [f"[{call.name} -> {status}]"]
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr}")
    return "\n".join(parts)
