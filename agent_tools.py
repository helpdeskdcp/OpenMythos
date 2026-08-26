"""
Concrete tool implementations for the local coding agent (Phase 8).

Every tool returns real subprocess/filesystem results (stdout, stderr, exit
code) -- never fabricated. All paths are resolved relative to a
`project_root` and refuse to escape it (no `..` traversal outside root), so
the agent can't be tricked into reading or writing outside the project it
was pointed at.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolResult:
    stdout: str
    stderr: str
    exit_code: int
    ok: bool


def _resolve(project_root: Path, rel_path: str) -> Path:
    """Resolve rel_path under project_root; raise if it would escape root."""
    root = project_root.resolve()
    target = (root / rel_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path {rel_path!r} escapes project root {root}")
    return target


def read_file(project_root: Path, path: str, max_bytes: int = 200_000) -> ToolResult:
    try:
        p = _resolve(project_root, path)
        if not p.is_file():
            return ToolResult("", f"not a file: {path}", 1, False)
        data = p.read_bytes()[:max_bytes]
        return ToolResult(data.decode("utf-8", errors="replace"), "", 0, True)
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def list_files(project_root: Path, path: str = ".", pattern: str = "*") -> ToolResult:
    try:
        root = project_root.resolve()
        p = _resolve(project_root, path)
        if not p.is_dir():
            return ToolResult("", f"not a directory: {path}", 1, False)
        entries = sorted(
            str(x.relative_to(root)) for x in p.rglob(pattern) if x.is_file()
        )
        return ToolResult("\n".join(entries), "", 0, True)
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def search_code(project_root: Path, query: str, glob: str = "*.py") -> ToolResult:
    try:
        proc = subprocess.run(
            ["grep", "-rn", "--include", glob, query, "."],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # grep exit code 1 means "no matches" -- not an error.
        return ToolResult(proc.stdout, proc.stderr, proc.returncode, proc.returncode in (0, 1))
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def run_python(project_root: Path, code: str = "", file: str = "", timeout: int = 30) -> ToolResult:
    try:
        if file:
            p = _resolve(project_root, file)
            cmd = ["python3", str(p)]
        elif code:
            cmd = ["python3", "-c", code]
        else:
            return ToolResult("", "run_python requires 'code' or 'file'", 1, False)
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=timeout)
        return ToolResult(proc.stdout, proc.stderr, proc.returncode, proc.returncode == 0)
    except subprocess.TimeoutExpired:
        return ToolResult("", f"timed out after {timeout}s", 1, False)
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def run_pytest(project_root: Path, args: str = "", timeout: int = 120) -> ToolResult:
    try:
        cmd = ["python3", "-m", "pytest"] + (args.split() if args else [])
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=timeout)
        return ToolResult(proc.stdout, proc.stderr, proc.returncode, proc.returncode == 0)
    except subprocess.TimeoutExpired:
        return ToolResult("", f"timed out after {timeout}s", 1, False)
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def compile_check(project_root: Path, file: str) -> ToolResult:
    try:
        p = _resolve(project_root, file)
        proc = subprocess.run(
            ["python3", "-m", "py_compile", str(p)],
            cwd=str(project_root), capture_output=True, text=True, timeout=30,
        )
        return ToolResult(proc.stdout, proc.stderr, proc.returncode, proc.returncode == 0)
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def apply_patch(project_root: Path, path: str, new_content: str) -> ToolResult:
    """Overwrite `path` with `new_content`. The caller (tool_router, gated by
    safety_policy) is responsible for approval before this is invoked -- this
    function only performs the write and reports what actually happened."""
    try:
        p = _resolve(project_root, path)
        old_len = len(p.read_text()) if p.exists() else 0
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new_content)
        return ToolResult(
            f"wrote {len(new_content)} bytes to {path} (was {old_len} bytes)", "", 0, True
        )
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def git_status(project_root: Path) -> ToolResult:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"], cwd=str(project_root),
            capture_output=True, text=True, timeout=15,
        )
        return ToolResult(proc.stdout, proc.stderr, proc.returncode, proc.returncode == 0)
    except Exception as e:
        return ToolResult("", str(e), 1, False)


def git_diff(project_root: Path, path: str = "") -> ToolResult:
    try:
        cmd = ["git", "diff"] + ([path] if path else [])
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=15)
        return ToolResult(proc.stdout, proc.stderr, proc.returncode, proc.returncode == 0)
    except Exception as e:
        return ToolResult("", str(e), 1, False)


TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "search_code": search_code,
    "run_python": run_python,
    "run_pytest": run_pytest,
    "compile_check": compile_check,
    "apply_patch": apply_patch,
    "git_status": git_status,
    "git_diff": git_diff,
}
