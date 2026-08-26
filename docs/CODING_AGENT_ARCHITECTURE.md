# Coding Agent Architecture

This is the Phase 8-13 implementation that follows `OPENMYTHOS_CODING_READINESS.md`.
That report found OpenMythos's from-scratch checkpoint fails Gates 1-3
(incoherent even under greedy decoding, no instruction tuning, no code
training data) and explicitly recommended **not** building an agent layer on
top of it — doing so would hide a model-quality problem behind a wrapper.

Instead, this agent is built on **`qwen2.5-coder:3b`**, already running
locally via Ollama on this host (`ollama list` showed it alongside
`phi3`, `gemma2:2b`, `llama3`, `tinyllama` — `qwen2.5-coder:3b` was picked
because it's the only code-specialized one). Confirmed directly
(not assumed) to produce correct, well-explained Python for a basic prompt,
with a real 32,768-token context window (vs. OpenMythos's 256).

## Why Ollama instead of OpenMythos

| | OpenMythos (this project's checkpoint) | qwen2.5-coder:3b (Ollama) |
|---|---|---|
| Instruction-tuned | No — 100% base/pretrain | Yes |
| Code training data | None found in either available corpus | Yes (code-specialized) |
| Context window | 256 tokens | 32,768 tokens |
| Gate 1 (coherent conversation) | FAIL (measured) | Pass (spot-checked) |
| Gate 2 (correct Python syntax) | FAIL (0/10 measured) | Pass (spot-checked) |

The agent layer needs a model that can reliably follow "read this file, find
the bug, propose a minimal patch" — that's an SFT/instruction-following
capability OpenMythos doesn't have yet. Nothing here is OpenMythos-specific;
swapping in a future, properly-trained OpenMythos checkpoint just means
pointing `mythos_cli.py --model` elsewhere (or writing an `OpenMythos`-backed
equivalent of `ollama_chat()`) once it passes Gates 1-4.

## Layout

```
mythos_cli.py       Chat CLI — the only file with a REPL / user-facing I/O
context_manager.py  Conversation + tool-result history, with truncation
tool_router.py      Parses a model's tool-call JSON, applies safety_policy,
                     dispatches to agent_tools, returns a real result string
agent_tools.py       The actual tool implementations (subprocess/filesystem)
safety_policy.py     SAFE vs REQUIRES_APPROVAL classification
```

```
User <-> mythos_cli.py <-> ContextManager (history, truncation)
              |
              v
        Ollama /api/chat  (qwen2.5-coder:3b)
              |
              v (assistant reply)
        tool_router.parse_tool_call()
         /                        \
   not a tool call            is a tool call
        |                            |
   print to user          safety_policy.classify_tool()
                            /                    \
                          SAFE            REQUIRES_APPROVAL
                           |                       |
                    agent_tools.TOOLS[name]   ask user y/N
                           |                       |
                    real stdout/stderr/exit   approved? run it : "[denied]"
                           |___________________________|
                                       |
                          fed back into ContextManager
                          as the next message, loop continues
```

## Tool-call protocol

Ollama's native OpenAI-style tool-calling (`tools=[...]` request param,
`message.tool_calls` response field) was tested directly against
`qwen2.5-coder:3b` on this host and came back as a plain JSON string in
`message.content` instead of a populated `tool_calls` array — so
`tool_router.parse_tool_call()` doesn't depend on that. Instead, the system
prompt asks the model to reply with exactly one JSON object,
`{"tool": "<name>", "args": {...}}`, when it wants to call a tool, and plain
text otherwise. `parse_tool_call()` also accepts the
`{"name": ..., "arguments": ...}` shape (what Ollama's template actually
produced), so it works either way regardless of which convention a given
model follows.

**Never fabricated**: every branch of `tool_router.execute_tool_call()` —
unknown tool, bad arguments, denied approval, or the tool itself failing —
returns an explicit, real string describing what happened, and that string
(not a guess) is what gets fed back to the model as the next message. The
model cannot see "success" unless the tool actually ran and reported one.

## Safety (Phase 9)

`safety_policy.classify_tool()` is what `tool_router` actually uses:

- **SAFE** (never asks): `read_file`, `list_files`, `search_code`,
  `run_pytest`, `compile_check`, `git_status`, `git_diff`
- **REQUIRES_APPROVAL** (CLI shows the exact call and asks `y/N` before
  running): `run_python` (arbitrary code execution), `apply_patch` (file
  mutation), and any unrecognized tool name (never assume safe by default)

`safety_policy.classify_shell_command()` is a second, independent
classifier over raw shell strings (not currently wired into `mythos_cli.py`,
since the agent's tool set is fixed rather than free-form shell) kept here
so a future direct-shell-exec tool reuses the same policy rather than
reinventing it. It matches the spec's own SAFE list (`pwd`, `ls`, `find`,
`grep`, `cat`, `python --version`, `python -m py_compile`, `pytest`,
`git status`, `git diff`) and REQUIRES_APPROVAL list (`rm`, `git reset`,
destructive `git checkout`/`clean`, package installs, service/system
commands, network commands, DB `DROP`/`DELETE`) — and REQUIRES_APPROVAL
patterns are checked first, so a destructive command chained after a safe
prefix (e.g. `git status && rm -rf .`) can't slip through.

## Context management (Phase 12)

`context_manager.ContextManager` holds a fixed system prompt plus a rolling
list of user/assistant/tool messages, and truncates from the **oldest**
message once a rough token estimate (`len(text)//4`) exceeds
`max_context_tokens - reserve_for_reply`. Tool results are folded into
`user`-role messages tagged `[tool result: <name>]` (most local chat APIs
don't have a first-class `tool` role for arbitrary models). This is
deliberately a rough heuristic, not exact token accounting — good enough to
keep prompts bounded without ever dumping a whole repository into one call;
targeted retrieval (`read_file`/`search_code` on demand) is the mechanism
for getting relevant code into context, not bulk inclusion.

## Debug loop (Phase 10)

Implemented as the `for _ in range(args.max_tool_turns)` loop in
`mythos_cli.py::main()`: each user turn lets the model chain up to
`--max-tool-turns` (default 6) tool calls — inspect a file, propose a patch,
compile-check it, run tests, read the failure, patch again — before either
producing a plain-text reply (stops the loop early) or hitting the turn cap
(prints `[stopped: too many chained tool calls this turn]` rather than
looping forever or silently giving up).

## Slash commands (Phase 11)

`/help /status /model /context /files /search <query> /test /compile <file>
/diff /reset /quit` are all handled directly in `mythos_cli.py` without a
model round-trip (they call the same `agent_tools.TOOLS` functions the model
itself uses), so a developer can inspect the project or run tests
immediately without needing the model to "decide" to do so first.

## What this does not cover yet

- Multi-file reasoning / repo-wide refactor planning: the tool set supports
  it (read/search across files), but there's no dedicated planner beyond the
  model's own tool-call sequencing — untested at that scale.
- `run_pytest`/`compile_check` are classified SAFE per the spec's own list,
  which assumes a **trusted** test suite; a test file could still execute
  arbitrary code. Worth revisiting if this is ever pointed at untrusted
  projects.
- No persistent session/task state across `mythos_cli.py` invocations —
  `/reset` and process exit both fully discard history today.
