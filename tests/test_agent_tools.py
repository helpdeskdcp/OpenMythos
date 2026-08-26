from pathlib import Path

import pytest

import agent_tools


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "foo.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "README.md").write_text("hello project\n")
    return tmp_path


def test_read_file_ok(project):
    r = agent_tools.read_file(project, "pkg/foo.py")
    assert r.ok
    assert "def add" in r.stdout


def test_read_file_missing(project):
    r = agent_tools.read_file(project, "nope.py")
    assert not r.ok
    assert r.exit_code == 1


def test_read_file_escape_blocked(project):
    r = agent_tools.read_file(project, "../../../../etc/passwd")
    assert not r.ok


def test_list_files(project):
    r = agent_tools.list_files(project, ".", "*.py")
    assert r.ok
    assert "pkg/foo.py" in r.stdout
    assert "README.md" not in r.stdout


def test_search_code_found(project):
    r = agent_tools.search_code(project, "def add")
    assert r.ok
    assert "foo.py" in r.stdout


def test_search_code_not_found(project):
    r = agent_tools.search_code(project, "nonexistent_symbol_xyz")
    assert r.ok  # grep exit 1 (no matches) is not a tool error
    assert r.stdout == ""


def test_run_python_code(project):
    r = agent_tools.run_python(project, code="print(1 + 1)")
    assert r.ok
    assert r.stdout.strip() == "2"


def test_run_python_file(project):
    r = agent_tools.run_python(project, file="pkg/foo.py")
    assert r.ok  # no error just defining a function


def test_run_python_syntax_error(project):
    r = agent_tools.run_python(project, code="def broken(:\n  pass")
    assert not r.ok
    assert r.exit_code != 0


def test_compile_check_valid(project):
    r = agent_tools.compile_check(project, "pkg/foo.py")
    assert r.ok


def test_compile_check_invalid(project):
    (project / "bad.py").write_text("def broken(:\n    pass\n")
    r = agent_tools.compile_check(project, "bad.py")
    assert not r.ok
    assert "SyntaxError" in r.stderr or "SyntaxError" in r.stdout


def test_apply_patch_writes_file(project):
    r = agent_tools.apply_patch(project, "pkg/foo.py", "def add(a, b):\n    return a + b\n")
    assert r.ok
    assert (project / "pkg" / "foo.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_apply_patch_creates_new_file(project):
    r = agent_tools.apply_patch(project, "new/dir/file.py", "x = 1\n")
    assert r.ok
    assert (project / "new" / "dir" / "file.py").read_text() == "x = 1\n"


def test_run_pytest_reports_real_failure(project):
    (project / "test_sample.py").write_text(
        "def test_fails():\n    assert 1 == 2\n"
    )
    r = agent_tools.run_pytest(project)
    assert not r.ok
    assert "1 failed" in r.stdout or "1 failed" in r.stderr


def test_run_pytest_reports_real_pass(project):
    (project / "test_sample.py").write_text(
        "def test_ok():\n    assert 1 == 1\n"
    )
    r = agent_tools.run_pytest(project)
    assert r.ok
    assert "1 passed" in r.stdout


def test_git_status_and_diff_on_non_repo(tmp_path):
    # Not a git repo: git commands fail with a real, non-fabricated error.
    r = agent_tools.git_status(tmp_path)
    assert not r.ok
    r2 = agent_tools.git_diff(tmp_path)
    assert not r2.ok
