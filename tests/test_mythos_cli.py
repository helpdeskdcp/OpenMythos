"""
Tests for the non-network parts of the mythos_cli agent stack:
tool-call parsing, safety classification, and context truncation. Does not
require a running Ollama server -- ollama_chat() itself (the HTTP call) is
intentionally left untested here since it needs a live model.
"""

from pathlib import Path

import agent_tools
from context_manager import ContextManager, estimate_tokens
from safety_policy import Safety, classify_shell_command, classify_tool
from tool_router import MalformedToolCall, ParsedToolCall, execute_tool_call, parse_tool_call


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------


def test_parse_tool_call_our_convention():
    call = parse_tool_call('{"tool": "read_file", "args": {"path": "x.py"}}')
    assert call is not None
    assert call.name == "read_file"
    assert call.args == {"path": "x.py"}


def test_parse_tool_call_ollama_native_shape():
    call = parse_tool_call('{"name": "list_files", "arguments": {"path": "."}}')
    assert call is not None
    assert call.name == "list_files"
    assert call.args == {"path": "."}


def test_parse_tool_call_plain_text_is_none():
    assert parse_tool_call("Sure, here's the function you asked for.") is None


def test_parse_tool_call_malformed_json_is_none():
    assert parse_tool_call('{"tool": "read_file", "args": ') is None


def test_parse_tool_call_json_without_tool_shape_is_none():
    # Valid JSON, but not a recognized tool-call shape -- should be treated
    # as a plain reply, not silently misinterpreted.
    assert parse_tool_call('{"hello": "world"}') is None


def test_parse_tool_call_args_as_bare_string_is_malformed_not_dropped():
    # Regression: observed live from qwen2.5-coder:3b -- args as a bare
    # string instead of a JSON object. Must be reported as an attempted,
    # malformed tool call (so the model can be told to retry), never
    # silently treated as `None` (which would show the raw JSON to the
    # user as if it were an ordinary reply).
    call = parse_tool_call('{"tool": "read_file", "args": "some/path.py"}')
    assert isinstance(call, MalformedToolCall)
    assert call.name == "read_file"
    assert "object" in call.error


def test_parse_tool_call_arguments_as_bare_string_is_malformed_not_dropped():
    call = parse_tool_call('{"name": "read_file", "arguments": "some/path.py"}')
    assert isinstance(call, MalformedToolCall)
    assert call.name == "read_file"


def test_parse_tool_call_missing_args_key_defaults_to_empty_dict():
    # Regression: observed live from qwen2.5-coder:3b -- {"tool": "list_files"}
    # with no "args" key at all. Many tools have all-optional arguments, so
    # this should be a valid call-with-defaults, not dropped as "not a tool
    # call" (which would show the raw JSON to the user as an ordinary reply).
    call = parse_tool_call('{"tool": "list_files"}')
    assert isinstance(call, ParsedToolCall)
    assert call.name == "list_files"
    assert call.args == {}


# ---------------------------------------------------------------------------
# safety_policy
# ---------------------------------------------------------------------------


def test_classify_tool_safe():
    assert classify_tool("read_file") is Safety.SAFE
    assert classify_tool("run_pytest") is Safety.SAFE
    assert classify_tool("git_diff") is Safety.SAFE


def test_classify_tool_requires_approval():
    assert classify_tool("run_python") is Safety.REQUIRES_APPROVAL
    assert classify_tool("apply_patch") is Safety.REQUIRES_APPROVAL


def test_classify_unknown_tool_defaults_to_approval():
    assert classify_tool("delete_everything") is Safety.REQUIRES_APPROVAL


def test_classify_shell_command_safe():
    assert classify_shell_command("git status") is Safety.SAFE
    assert classify_shell_command("git diff foo.py") is Safety.SAFE
    assert classify_shell_command("python3 -m pytest tests/") is Safety.SAFE
    assert classify_shell_command("ls -la") is Safety.SAFE


def test_classify_shell_command_requires_approval():
    assert classify_shell_command("rm -rf /") is Safety.REQUIRES_APPROVAL
    assert classify_shell_command("git reset --hard") is Safety.REQUIRES_APPROVAL
    assert classify_shell_command("pip install requests") is Safety.REQUIRES_APPROVAL
    assert classify_shell_command("systemctl restart nginx") is Safety.REQUIRES_APPROVAL


def test_classify_shell_command_dangerous_suffix_not_hidden_by_safe_prefix():
    # A destructive command chained after a safe-looking one must still be
    # caught -- approval patterns are checked first, everywhere in the string.
    assert classify_shell_command("git status && rm -rf .") is Safety.REQUIRES_APPROVAL


def test_classify_shell_command_unknown_defaults_to_approval():
    assert classify_shell_command("some_random_binary --flag") is Safety.REQUIRES_APPROVAL


# ---------------------------------------------------------------------------
# execute_tool_call (approval gating + real results, no fabrication)
# ---------------------------------------------------------------------------


def test_execute_tool_call_safe_tool_runs_without_asking(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    asked = []
    call = parse_tool_call('{"tool": "read_file", "args": {"path": "a.py"}}')
    out = execute_tool_call(tmp_path, call, lambda name, args: asked.append(name) or True)
    assert "x = 1" in out
    assert asked == []  # never consulted approval for a SAFE tool


def test_execute_tool_call_requires_approval_and_respects_denial(tmp_path: Path):
    call = parse_tool_call('{"tool": "apply_patch", "args": {"path": "a.py", "new_content": "y = 2\\n"}}')
    out = execute_tool_call(tmp_path, call, lambda name, args: False)
    assert "denied" in out.lower()
    assert not (tmp_path / "a.py").exists()  # denial must actually block the write


def test_execute_tool_call_requires_approval_and_respects_approval(tmp_path: Path):
    call = parse_tool_call('{"tool": "apply_patch", "args": {"path": "a.py", "new_content": "y = 2\\n"}}')
    out = execute_tool_call(tmp_path, call, lambda name, args: True)
    assert "OK" in out
    assert (tmp_path / "a.py").read_text() == "y = 2\n"


def test_execute_tool_call_unknown_tool_reports_error_not_success(tmp_path: Path):
    call = parse_tool_call('{"tool": "delete_everything", "args": {}}')
    out = execute_tool_call(tmp_path, call, lambda name, args: True)
    assert "unknown tool" in out.lower()


def test_execute_tool_call_bad_arguments_reports_error(tmp_path: Path):
    call = parse_tool_call('{"tool": "read_file", "args": {"nonexistent_kwarg": "x"}}')
    out = execute_tool_call(tmp_path, call, lambda name, args: True)
    assert "error" in out.lower()


def test_execute_tool_call_never_fabricates_a_failed_result(tmp_path: Path):
    call = parse_tool_call('{"tool": "read_file", "args": {"path": "missing.py"}}')
    out = execute_tool_call(tmp_path, call, lambda name, args: True)
    assert "FAILED" in out
    assert "not a file" in out


# ---------------------------------------------------------------------------
# context_manager
# ---------------------------------------------------------------------------


def test_context_manager_roundtrip_roles():
    ctx = ContextManager(system_prompt="sys")
    ctx.add_user("hello")
    ctx.add_assistant("hi there")
    msgs = ctx.build_messages()
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "hello"}
    assert msgs[2] == {"role": "assistant", "content": "hi there"}


def test_context_manager_tool_result_tagged_as_user_role():
    ctx = ContextManager(system_prompt="sys")
    ctx.add_tool_result("read_file", "file contents here")
    msgs = ctx.build_messages()
    assert msgs[-1]["role"] == "user"
    assert "[tool result: read_file]" in msgs[-1]["content"]


def test_context_manager_truncates_oldest_first():
    ctx = ContextManager(system_prompt="sys", max_context_tokens=50, reserve_for_reply=0)
    for i in range(20):
        ctx.add_user(f"message number {i} " * 5)
    msgs = ctx.build_messages()
    total = sum(estimate_tokens(m["content"]) for m in msgs)
    assert total <= 50
    # the most recent message must survive truncation
    assert "message number 19" in msgs[-1]["content"]
    # and the earliest ones must have been dropped
    assert not any("message number 0 " in m["content"] for m in msgs)


def test_context_manager_reset_clears_history():
    ctx = ContextManager(system_prompt="sys")
    ctx.add_user("hello")
    ctx.reset()
    assert ctx.messages == []
