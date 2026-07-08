"""Test execution harness for EvalPlus suites.

Runs candidate solutions against EvalPlus-augmented test suites (HumanEval+ /
MBPP+) in an isolated subprocess with a timeout.

Carries forward the spike's key fix: strip model-injected self-test assertions
from instruct-model outputs before grading. Qwen2.5-Instruct appends its own
``assert`` blocks, which pollute test results if left in place.

Status: NOT IMPLEMENTED — Phase 1 begins after the Phase 0 decision gates
(model + benchmark confirmation) are closed. See spec.md.
"""
