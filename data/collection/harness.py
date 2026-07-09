"""Test execution harness for EvalPlus suites.

This module owns everything between "raw model output" and "graded result":
extracting code from chat responses, stripping model-injected self-tests, and
(to come) running candidates against EvalPlus-augmented test suites in an
isolated subprocess.

The stripping step is load-bearing: instruct models (confirmed twice for
Qwen2.5-1.5B-Instruct, in the feasibility spike and the n=25 run) append their
own ``assert`` blocks after the solution. Left in place, those self-tests are
graded alongside the benchmark's tests and corrupt pass/fail labels.

Extraction logic is ported from the validated ``spike/spike.py``.

PLATFORM CONSTRAINT (discovered 2026-07-08, see RESEARCH_LOG.md): EvalPlus
grading only works on POSIX systems. Its sandbox uses the Unix-only
``resource`` module and ``signal.setitimer``; on Windows the grader child
process crashes and every result comes back as a bogus "timeout". ``grade``
therefore refuses to run on Windows — use WSL locally or run on Linux (Colab).
EvalPlus imports are kept inside functions so that the pure text utilities
work everywhere without the heavy dependency chain.
"""

import os
import re
from dataclasses import dataclass

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


def prepare_solution(raw_generation: str) -> str:
    """Turn a raw chat generation into a gradeable candidate solution.

    Composes the two text-cleaning steps: pull the fenced code block, then
    strip trailing self-tests. This is the single call sites should use so
    the cleaning pipeline stays consistent everywhere.

    Args:
        raw_generation: Full text of the model's chat response.

    Returns:
        Candidate solution code ready for grading.
    """
    return strip_self_tests(extract_code_block(raw_generation))


# --------------------------------------------------------------------------- #
# EvalPlus grading (POSIX only — see module docstring)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GradeResult:
    """Outcome of grading one candidate against one EvalPlus problem.

    Attributes:
        base_status: EvalPlus status ("pass"/"fail"/"timeout") on the original
            HumanEval/MBPP test inputs.
        plus_status: Status on the EvalPlus-augmented inputs — the stricter
            suite that catches spurious passes.
        passed: True only if BOTH suites pass. This is the label-relevant
            definition of success for the whole project: a base-only pass is
            exactly the label noise EvalPlus exists to remove.
    """

    base_status: str
    plus_status: str

    @property
    def passed(self) -> bool:
        """Whether the candidate passed both the base and plus suites."""
        return self.base_status == "pass" and self.plus_status == "pass"


def load_problem_suite(subset: str) -> tuple[dict, dict]:
    """Load EvalPlus problems and groundtruth expected outputs.

    Mirrors the calls ``evalplus.evaluate.evaluate`` itself makes, including
    the dataset-hash cache key, so groundtruth computation happens once and
    is reused from EvalPlus's own cache directory.

    Args:
        subset: "humaneval" or "mbpp".

    Returns:
        A tuple ``(problems, expected_output)``: task_id-keyed problem dicts,
        and task_id-keyed expected outputs for base and plus inputs.

    Raises:
        ValueError: If ``subset`` is not a supported EvalPlus subset.
    """
    if subset not in ("humaneval", "mbpp"):
        raise ValueError(f"unknown EvalPlus subset: {subset!r}")

    from evalplus.data import (
        get_human_eval_plus,
        get_human_eval_plus_hash,
        get_mbpp_plus,
        get_mbpp_plus_hash,
    )
    from evalplus.eval import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    if subset == "humaneval":
        problems = get_human_eval_plus()
        expected = get_groundtruth(problems, get_human_eval_plus_hash(), [])
    else:
        problems = get_mbpp_plus()
        expected = get_groundtruth(
            problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS
        )
    return problems, expected


def grade(subset: str, problem: dict, expected_output: dict, solution: str) -> GradeResult:
    """Grade one candidate solution against base and plus test suites.

    Args:
        subset: "humaneval" or "mbpp" (EvalPlus needs it for output rules).
        problem: One problem dict from ``load_problem_suite``.
        expected_output: That problem's entry from the expected-output dict.
        solution: Candidate code, already cleaned by ``prepare_solution``.

    Returns:
        A GradeResult with per-suite statuses.

    Raises:
        RuntimeError: On Windows, where EvalPlus's sandbox silently breaks
            (Unix-only ``resource``/``signal.setitimer``) and would return
            fake "timeout" results for every solution.
    """
    if os.name == "nt":
        raise RuntimeError(
            "EvalPlus grading does not work on Windows: its sandbox needs "
            "Unix-only APIs and misreports every result as 'timeout'. "
            "Run grading under WSL or on Linux (e.g. the Colab collector)."
        )
    from evalplus.evaluate import check_correctness

    result = check_correctness(subset, 0, problem, solution, expected_output)
    return GradeResult(
        base_status=result["base"][0],
        plus_status=result["plus"][0],
    )
