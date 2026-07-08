"""Activation extraction via TransformerLens hooks.

Captures residual stream, per-head outputs, and MLP outputs at every layer
during generation, using ``model.run_with_hooks()`` with a ``names_filter``
for efficiency.

Status: NOT IMPLEMENTED — Phase 2 begins after the labeled dataset meets its
exit criteria (>=50 examples per failure class, spot-check accuracy >=85%).
"""
