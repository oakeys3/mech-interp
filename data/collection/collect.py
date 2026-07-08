"""Main dataset collection script.

Batch generation with the primary model over EvalPlus problems, with a
multi-round self-correction loop (up to 3 correction rounds per problem).
Every attempt is logged to JSONL under ``data/raw/`` with a version-controlled
schema.

Status: NOT IMPLEMENTED — Phase 1 begins after the Phase 0 decision gates
(model + benchmark confirmation) are closed. See spec.md.
"""
