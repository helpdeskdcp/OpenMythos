#!/usr/bin/env python3
"""
Local, Ollama-backed developer CLI (Phase 11) -- a Claude-Code-like local
workflow, built on a real capable local model rather than the from-scratch
OpenMythos checkpoint, which per OPENMYTHOS_CODING_READINESS.md fails
Gates 1-3 today (incoherent even under greedy decoding, no instruction
tuning, no code training data). Default model is qwen2.5-coder:3b, already
pulled in this host's Ollama instance and confirmed to produce correct,
well-explained Python.

    python mythos_cli.py
    python mythos_cli.py --project-root /path/to/some/project --model qwen2.5-coder:3b

Commands: /help /status /model /context /files /search <query> /test
/compile <file> /diff /reset /quit
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from agent_tools import TOOLS
from context_manager import ContextManager
from tool_router import MalformedToolCall, execute_tool_call, parse_tool_call

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT_TEMPLATE = """You are a local Python coding and debugging \
assistant working on a real project on disk at {root}.

You have these tools. To use one, respond with ONLY a single JSON object \
of the form {{"tool": "<name>", "args": {{...}}}} and nothing else -- no \
markdown fences, no extra prose around it. Otherwise, respond in plain \
text to the user.

Tools:
- read_file(path): read a file's contents
- list_files(path=".", pattern="*"): list files under a directory
- search_code(query, glob="*.py"): grep for a string in the project
- run_pytest(args=""): run the project's test suite
- compile_check(file): python -m py_compile a file (syntax check only)
- run_python(code="", file="", timeout=30): execute Python (requires approval)
- apply_patch(path, new_content): overwrite a file with new content (requires approval)
- git_status(): git status --short
- git_diff(path=""): git diff, optionally scoped to one path

Debugging workflow: understand the task, inspect relevant files first \
(read_file/search_code) before proposing anything, form a hypothesis, make \
the smallest patch that could fix it (never rewrite a whole file when a \
small patch will do), run compile_check then run_pytest, read the actual \
failure output, and iterate. When a test fails, fix the implementation to \
satisfy the test's stated intent -- do not edit the test's expected value \
just to make it pass, unless you have clear evidence the test itself is \
wrong (e.g. it contradicts the task description). Never claim a command \
ran or a file changed unless you actually called the corresponding tool \
and are looking at its real result. If you're blocked, say so plainly \
instead of guessing.
"""


def ollama_chat(model: str, messages: list[dict], timeout: int = 180) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def cli_approval_hook(tool_name: str, args: dict) -> bool:
    print(f"\n[approval required] {tool_name}({json.dumps(args)})")
    ans = input("  Allow this action? [y/N] ").strip().lower()
    return ans == "y"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--project-root", default=".")
    p.add_argument("--model", default="qwen2.5-coder:3b")
    p.add_argument(
        "--max-tool-turns",
        type=int,
        default=6,
        help="max chained tool calls per user turn before forcing a stop",
    )
    p.add_argument("--max-context-tokens", type=int, default=6000)
    return p.parse_args()


def print_help() -> None:
    print(
        "/help                show this message\n"
        "/status              model + project root + message count\n"
        "/model               current model name\n"
        "/context             dump the rolling conversation history\n"
        "/files               list project files\n"
        "/search <query>      grep the project for <query>\n"
        "/test                run the project's pytest suite\n"
        "/compile <file>      python -m py_compile a file\n"
        "/diff                git diff for the project\n"
        "/reset               clear conversation history\n"
        "/quit                exit"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    ctx = ContextManager(
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(root=root),
        max_context_tokens=args.max_context_tokens,
    )

    print(f"[mythos_cli] model={args.model}  project_root={root}")
    print("Type a request, or /help for commands.\n")

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue

        if user_text in ("/quit", "/exit"):
            break
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/status":
            print(f"model={args.model} project_root={root} messages={len(ctx.messages)}")
            continue
        if user_text == "/model":
            print(args.model)
            continue
        if user_text == "/context":
            for m in ctx.messages:
                print(f"  [{m['role']}] {m['content'][:80]!r}")
            continue
        if user_text == "/files":
            r = TOOLS["list_files"](root, ".")
            print(r.stdout if r.ok else r.stderr)
            continue
        if user_text.startswith("/search "):
            r = TOOLS["search_code"](root, user_text[len("/search "):])
            print(r.stdout if r.stdout else "(no matches)")
            continue
        if user_text == "/test":
            r = TOOLS["run_pytest"](root)
            print(r.stdout or r.stderr)
            continue
        if user_text.startswith("/compile "):
            r = TOOLS["compile_check"](root, user_text[len("/compile "):])
            print(r.stdout or r.stderr or "OK: compiles cleanly")
            continue
        if user_text == "/diff":
            r = TOOLS["git_diff"](root)
            print(r.stdout or "(no changes)")
            continue
        if user_text == "/reset":
            ctx.reset()
            print("[context cleared]")
            continue

        ctx.add_user(user_text)

        for _ in range(args.max_tool_turns):
            try:
                reply = ollama_chat(args.model, ctx.build_messages())
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                print(f"[error] could not reach Ollama at {OLLAMA_CHAT_URL}: {e}")
                break

            call = parse_tool_call(reply)
            if call is None:
                ctx.add_assistant(reply)
                print(reply)
                break

            if isinstance(call, MalformedToolCall):
                error_text = f"[tool error] {call.error}"
                print(error_text)
                ctx.add_assistant(reply)
                ctx.add_tool_result(call.name, error_text)
                continue  # let the model retry with corrected args, same turn budget

            print(f"[tool call] {call.name}({call.args})")
            result_text = execute_tool_call(root, call, cli_approval_hook)
            print(result_text)
            ctx.add_assistant(reply)
            ctx.add_tool_result(call.name, result_text)
        else:
            print("[stopped: too many chained tool calls this turn]")


if __name__ == "__main__":
    main()
