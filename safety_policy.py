"""
Command/tool safety classification (Phase 9).

Two independent classifiers:
  classify_tool(name)          -- for the agent's own fixed tool set
                                   (agent_tools.py)
  classify_shell_command(cmd)  -- for a raw shell command string, kept here
                                   so any future direct shell-exec tool
                                   reuses the same policy as the fixed tools

Unknown input always defaults to REQUIRES_APPROVAL -- never SAFE by default.
"""

from __future__ import annotations

import re
from enum import Enum


class Safety(Enum):
    SAFE = "safe"
    REQUIRES_APPROVAL = "requires_approval"


# Tools with no side effects beyond reading the filesystem or running the
# project's own test suite / syntax checker.
_SAFE_TOOLS = {
    "read_file",
    "list_files",
    "search_code",
    "run_pytest",
    "compile_check",
    "git_status",
    "git_diff",
}

# Tools that execute arbitrary code or mutate files.
_APPROVAL_TOOLS = {
    "run_python",
    "apply_patch",
}


def classify_tool(name: str) -> Safety:
    """Classify one of agent_tools.py's fixed tool names."""
    if name in _SAFE_TOOLS:
        return Safety.SAFE
    if name in _APPROVAL_TOOLS:
        return Safety.REQUIRES_APPROVAL
    return Safety.REQUIRES_APPROVAL  # unknown tool: never assume safe


# Matched against a raw shell command string, in order. A REQUIRES_APPROVAL
# match always wins over a SAFE match (e.g. "git diff && rm -rf /" must not
# slip through because it starts with "git diff").
_APPROVAL_PATTERNS = [
    r"\brm\s", r"^rm\b", r"^del\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+checkout\s+--\b", r"\bgit\s+checkout\s+\.\b",
    r"\bgit\s+clean\b",
    r"\bpip3?\s+install\b", r"\bapt(-get)?\s+install\b", r"\bnpm\s+install\b",
    r"\bsystemctl\b", r"\bservice\s+\w+\s+(restart|stop|start)\b",
    r"\breboot\b", r"\bshutdown\b",
    r"\bcurl\b", r"\bwget\b",
    r"\bdrop\s+table\b", r"\bdelete\s+from\b", r"\btruncate\s+table\b",
]

_SAFE_PATTERNS = [
    r"^pwd\b", r"^ls\b", r"^dir\b", r"^find\b", r"^grep\b", r"^cat\b",
    r"^python3?\s+--version\b",
    r"^python3?\s+-m\s+py_compile\b",
    r"^pytest\b", r"^python3?\s+-m\s+pytest\b",
    r"^git\s+status\b", r"^git\s+diff\b",
]


def classify_shell_command(cmd: str) -> Safety:
    """Classify a raw shell command string. Approval patterns take priority
    over safe patterns so a destructive command can't hide behind a safe
    prefix (e.g. chained with `&&`)."""
    stripped = cmd.strip()
    for pat in _APPROVAL_PATTERNS:
        if re.search(pat, stripped, re.IGNORECASE):
            return Safety.REQUIRES_APPROVAL
    for pat in _SAFE_PATTERNS:
        if re.match(pat, stripped, re.IGNORECASE):
            return Safety.SAFE
    return Safety.REQUIRES_APPROVAL  # unrecognized command: default to approval
