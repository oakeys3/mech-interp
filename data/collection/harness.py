"""Test execution harness for EvalPlus suites.

This module owns everything between "raw model output" and "graded result":
extracting code from chat responses, stripping model-injected self-tests, and
(to come) running candidates against EvalPlus-augmented test suites in an
isolated subprocess.

The stripping step is load-bearing: instruct models (confirmed twice for
Qwen2.5-1.5B-Instruct, in the feasibility spike and the n=25 run) append their
own ``assert`` blocks after the solution. Left in place, those self-tests are
graded alongside the benchmark's tests and corrupt pass/fail labels.

Ported from the validated logic in ``spike/spike.py``. The EvalPlus runner is
not implemented yet — see RESEARCH_LOG.md (Phase 1 plan).
"""

import re

# Test-scaffolding markers that end the solution portion of a generation.
# Each starts with "\n" followed immediately by the keyword, so they match
# only at column zero: top-level self-tests are cut, while indented asserts
# inside a function body (legitimate input validation) are preserved.
_SELF_TEST_MARKERS = ["\nassert ", "\nif __name__", "\n# Test", "\n# test", "\nprint("]

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Extract the first fenced code block from a chat-model response.

    Instruct models are prompted to return the solution in a ```python fence.
    If no fence is found, the whole text is returned unchanged so downstream
    stripping and grading still get a chance to run.

    Args:
        text: Raw generation from the model, typically markdown-ish chat text.

    Returns:
        The contents of the first fenced code block, or ``text`` verbatim if
        no fence is present.
    """
    match = _CODE_FENCE.search(text)
    return match.group(1) if match else text


def strip_self_tests(code: str) -> str:
    """Cut model-appended self-tests from the end of a candidate solution.

    Truncates at the earliest top-level test-scaffolding line: a bare
    ``assert``, an ``if __name__`` guard, a ``# Test``/``# test`` comment, or
    a top-level ``print(``. Everything before the earliest marker is kept,
    including imports and helper functions defined before the self-tests.

    Known limitation (accepted): if the model interleaves self-tests *between*
    function definitions, any definitions after the first marker are lost.
    Not observed in practice — Qwen appends tests strictly after the solution.

    Args:
        code: Candidate solution code, already extracted from any code fence.

    Returns:
        The solution with trailing self-test scaffolding removed and trailing
        whitespace stripped.
    """
    cut = len(code)
    for marker in _SELF_TEST_MARKERS:
        index = code.find(marker)
        if index != -1:
            cut = min(cut, index)
    return code[:cut].rstrip()
