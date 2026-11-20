# Instructions for GitHub Copilot

The guidelines of this repository are written once, for every coding agent, in
[AGENTS.md](../AGENTS.md) and in the guides it links to:

- [docs/development/workflow.md](../docs/development/workflow.md): Conventional Commits,
  issue and pull request templates, mandatory labels, branches, releases;
- [docs/development/architecture.md](../docs/development/architecture.md): Rust core, PyO3
  bindings, Python API, their invariants, documentation in the code and performance;
- [docs/development/testing.md](../docs/development/testing.md): test-driven development,
  expected values from formulas and papers;
- [docs/development/security.md](../docs/development/security.md): secure coding, dependencies,
  CI and secrets.

Follow them for every suggestion, review and change. In short: Conventional Commits for
commits, pull request titles and branch names; always use the issue and pull request
templates and label both; write the failing test before the code; document Python with
docstrings and keep inline comments for particular logic only; formulas come from a cited paper and
live in the Rust core; never weaken a check to make a change pass.

To review a change, follow the procedure in
[.agents/skills/review/SKILL.md](../.agents/skills/review/SKILL.md) and the review rules
in [AGENTS.md](../AGENTS.md#review).
