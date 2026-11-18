"""
Test suite of recsyslearn.

Hypothesis runs with derandomize=True, so property-based tests generate the same examples on every
run and CI stays deterministic (see AGENTS.md, section 4).
"""

from hypothesis import settings

settings.register_profile("recsyslearn", derandomize=True, deadline=None)
settings.load_profile("recsyslearn")
