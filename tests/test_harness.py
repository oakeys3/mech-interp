"""Unit tests for harness, labeling, and probe shapes."""

from pathlib import Path

import yaml

import os

import pytest

from data.collection.harness import (
    GradeResult,
    extract_code_block,
    grade,
    load_problem_suite,
    prepare_solution,
    strip_self_tests,
)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "experiment_config.yaml"


def load_config() -> dict:
    """Load the experiment config from its canonical location.

    Returns:
        The parsed YAML config as a dict.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_exists_and_parses() -> None:
    """The experiment config must exist and be valid YAML."""
    config = load_config()
    assert isinstance(config, dict)


def test_config_has_required_sections() -> None:
    """All top-level sections referenced by the pipeline must be present."""
    config = load_config()
    for section in ("model", "benchmark", "generation", "probing", "patching", "labels"):
        assert section in config, f"missing config section: {section}"


def test_label_taxonomy_is_complete() -> None:
    """The label set must be exactly the P/F1-F4 per-stage taxonomy."""
    config = load_config()
    assert config["labels"]["classes"] == ["P", "F1", "F2", "F3", "F4"]


def test_generation_settings_are_sane() -> None:
    """Generation settings must stay within the ranges the spike validated."""
    gen = load_config()["generation"]
    assert 0.0 <= gen["temperature"] <= 1.0
    assert gen["correction_rounds"] >= 1
    assert gen["max_new_tokens"] > 0


# --------------------------------------------------------------------------- #
# extract_code_block
# --------------------------------------------------------------------------- #

def test_extract_fenced_python_block() -> None:
    """A ```python fence yields exactly its contents."""
    text = "Here is the solution:\n```python\ndef f(x):\n    return x\n```\nDone."
    assert extract_code_block(text) == "def f(x):\n    return x\n"


def test_extract_fence_without_language_tag() -> None:
    """A bare ``` fence works the same as a ```python fence."""
    text = "```\ndef f(x):\n    return x\n```"
    assert extract_code_block(text) == "def f(x):\n    return x\n"


def test_extract_without_fence_returns_input() -> None:
    """No fence means the text is passed through for downstream handling."""
    text = "def f(x):\n    return x"
    assert extract_code_block(text) == text


def test_extract_takes_first_of_multiple_blocks() -> None:
    """Documents current behavior: the FIRST block wins.

    Correction responses could in principle quote old code first; if that is
    observed in Phase 1 data, this choice must be revisited.
    """
    text = "```python\nfirst = 1\n```\ntext\n```python\nsecond = 2\n```"
    assert extract_code_block(text) == "first = 1\n"


# --------------------------------------------------------------------------- #
# strip_self_tests
# --------------------------------------------------------------------------- #

def test_strip_trailing_asserts() -> None:
    """Top-level asserts appended after the solution are removed."""
    code = "def f(x):\n    return x\n\nassert f(1) == 1\nassert f(2) == 2\n"
    assert strip_self_tests(code) == "def f(x):\n    return x"


def test_strip_preserves_indented_assert() -> None:
    """An assert inside a function body is legitimate code, not a self-test."""
    code = "def f(x):\n    assert x > 0\n    return x"
    assert strip_self_tests(code) == code


def test_strip_test_comment_and_main_guard() -> None:
    """`# Test cases` comments and `if __name__` guards both cut the code."""
    code = "def f(x):\n    return x\n\n# Test cases\nprint(f(1))\n"
    assert strip_self_tests(code) == "def f(x):\n    return x"
    code = "def f(x):\n    return x\n\nif __name__ == '__main__':\n    f(1)\n"
    assert strip_self_tests(code) == "def f(x):\n    return x"


def test_strip_keeps_imports_and_helpers() -> None:
    """Everything before the first marker survives, including helper defs."""
    code = (
        "from typing import List\n\n"
        "def helper(x):\n    return x + 1\n\n"
        "def f(xs: List[int]):\n    return [helper(x) for x in xs]\n\n"
        "assert f([1]) == [2]\n"
    )
    expected = (
        "from typing import List\n\n"
        "def helper(x):\n    return x + 1\n\n"
        "def f(xs: List[int]):\n    return [helper(x) for x in xs]"
    )
    assert strip_self_tests(code) == expected


def test_strip_noop_when_clean() -> None:
    """Code without self-tests is returned unchanged (modulo trailing ws)."""
    code = "def f(x):\n    return x"
    assert strip_self_tests(code) == code


def test_full_pipeline_on_observed_qwen_output() -> None:
    """Regression fixture shaped like Qwen's gen2 for HumanEval/1 (n=25 run):
    fenced block with the solution followed by appended self-test asserts."""
    raw = (
        "```python\n"
        "from typing import *\n\n"
        "def separate_paren_groups(paren_string: str) -> List[str]:\n"
        "    stack = []\n"
        "    return stack\n\n"
        "# Test cases\n"
        'assert separate_paren_groups("( )") == ["()"]\n'
        "```"
    )
    code = prepare_solution(raw)
    assert code.endswith("return stack")
    assert "assert" not in code
    assert "# Test" not in code
    assert code == strip_self_tests(extract_code_block(raw))


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def test_grade_result_requires_both_suites() -> None:
    """A base-only pass is a spurious pass, not a pass."""
    assert GradeResult("pass", "pass").passed
    assert not GradeResult("pass", "fail").passed
    assert not GradeResult("fail", "pass").passed
    assert not GradeResult("timeout", "timeout").passed


def test_load_problem_suite_rejects_unknown_subset() -> None:
    """Subset validation happens before any EvalPlus import."""
    with pytest.raises(ValueError, match="unknown EvalPlus subset"):
        load_problem_suite("apps")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only guard")
def test_grade_refuses_windows() -> None:
    """On Windows, grading must fail loudly, never return bogus timeouts."""
    with pytest.raises(RuntimeError, match="Windows"):
        grade("humaneval", {}, {}, "def f(): pass")


@pytest.mark.skipif(
    os.name == "nt", reason="EvalPlus sandbox is POSIX-only"
)
def test_grade_canonical_and_broken_solutions() -> None:
    """Integration: the canonical solution passes, a broken one fails.

    Skipped on Windows; requires evalplus (skipped if not installed, e.g. in
    the lightweight CI environment).
    """
    pytest.importorskip("evalplus")
    problems, expected = load_problem_suite("humaneval")
    task = "HumanEval/0"
    canonical = problems[task]["prompt"] + problems[task]["canonical_solution"]
    broken = problems[task]["prompt"] + "    return False\n"
    assert grade("humaneval", problems[task], expected[task], canonical).passed
    assert not grade("humaneval", problems[task], expected[task], broken).passed

