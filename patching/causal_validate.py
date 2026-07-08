"""Causal validation: patch -> rerun -> measure failure-rate shift.

For each patching intervention, rerun problem generation and measure the
delta pass rate vs. baseline. Priority order: F2, F1, then F3/F4 as stretch.
Results are written as JSONL under ``patching/results/``.

Status: NOT IMPLEMENTED — Phase 3. See spec.md.
"""
