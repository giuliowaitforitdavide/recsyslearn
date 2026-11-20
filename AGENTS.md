# Guidelines for coding agents

This file is the entry point for any automated coding agent (Claude Code, GitHub Copilot,
Codex, Cursor, ...) working on recsyslearn. People follow the same rules, starting from
[CONTRIBUTING.md](CONTRIBUTING.md).

recsyslearn evaluates recommender systems (accuracy, beyond-accuracy and fairness metrics,
and segmentations of users and items). It is a Python package (`python/recsyslearn`) whose
computations run in a Rust core (`crates/recsyslearn-core`) exposed through PyO3
(`crates/recsyslearn-py`). Every metric is implemented from a published definition.

## Read before you change anything

| Guide | Read it when |
| --- | --- |
| [docs/development/workflow.md](docs/development/workflow.md) | Always: issues, branches, commits, pull requests, releases. |
| [docs/development/architecture.md](docs/development/architecture.md) | Before touching code: layers, data contract, invariants, documentation in the code, performance. |
| [docs/development/testing.md](docs/development/testing.md) | Before writing code: tests come first. |
| [docs/development/security.md](docs/development/security.md) | Always: what must never be done. |

## Rules that are never optional

1. **Conventional Commits** for every commit message, pull request title and branch name
   (`feat/...`, `fix/...`): `type(scope): description`. See the workflow guide for the types.
2. **Use the templates.** Open issues with a template from `.github/ISSUE_TEMPLATE`
   (`gh issue create --template "Bug report"`) and pull requests with
   `.github/pull_request_template.md` (`gh pr create --body-file ...`, then fill it in).
   Fill in every section and keep every checklist item; never replace a template with
   free text.
3. **Label everything.** Every issue and pull request gets at least one of the existing
   labels when it is opened, passed explicitly with `--label`; never create a label. The
   mapping is in the workflow guide.
4. **Tests first.** Write a failing test from the definition, then the implementation.
   Expected values come from the formula or from a paper, never from the output of the
   code under test. A bug fix starts with a test that reproduces the bug.
5. **Science first.** A metric or segmentation is added or changed only from a published
   source, cited in the docstring and in `docs/methodology.rst`, with every choice the
   source leaves open documented. Never tune code or tests to make numbers agree.
6. **Respect the layers.** Formulas live in the Rust core, which has no `unsafe`, does not
   panic on user input and does no I/O; the bindings contain no logic; the Python layer
   validates, encodes identifiers and labels results, without per-row Python loops and
   without modifying its inputs.
7. **Docstrings, not comments.** Python is documented with docstrings and Rust with
   documentation comments (`///`, `//!`). Inline comments are only for particular logic
   and say why, never what; no commented-out code; tool directives name the rule and give
   a reason. See the architecture guide.
8. **Do not weaken a check to get a change through**: no skipped tests, no disabled lints,
   no removed `--locked`, no loosened assertions.
9. **Do not commit to `main`, push, publish or change repository settings** unless a
   maintainer asked for that action. Never commit secrets or generated files.

## Review

These rules apply to every review, by a person or by an agent.

- Every finding cites the rule it refers to (file and heading of a guide) or describes
  the concrete failure (an input and the wrong output, or the command that fails).
- Not reported: problems that existed before the change, anything Ruff, rustfmt, Clippy
  or CI already report, style preferences that no guide states, speculation.
- The author answers every Mountain and Boulder finding, by fixing it or by explaining
  why it does not apply.
- Reviews by agents are advisory. The final approval is always given by a human
  maintainer, and a change written by an agent is never approved by an agent alone.
- **Three rounds of agent review per pull request, each with its own scope.** A round is
  a review whose result is *published*, i.e. posted on the pull request, by anyone; a
  local run whose result stays with the author is not a round.

  | Round | What is reviewed |
  | --- | --- |
  | 1 | The whole diff of the open pull request: bugs and broken rules. |
  | 2 | Only the changes made in answer to the findings of round 1: is each Mountain and Boulder finding answered, and did the answers introduce new problems? |
  | 3 | Only the changes made in answer to the findings of round 2, in the same way. |

  A published review opens with `Agent review <n> of 3 at <commit>`, the commit it
  reviewed, so that the next round knows where its diff starts. After round 3 no agent
  reviews the pull request again, locally or not: a human reviewer takes over and decides
  on whatever is still open. The rounds belong to the pull request and are not reset by
  new commits.

### Severity scale

The scale is the [Netlify Feedback Ladder](https://www.netlify.com/blog/2020/03/05/feedback-ladders-how-we-encode-code-reviews-at-netlify/).

| Level | Blocks the merge | Meaning in recsyslearn |
| --- | --- | --- |
| Mountain | Yes, fix immediately | Wrong results; a formula that differs from the cited source; a panic, crash or memory-safety problem; an argument modified; a security problem. |
| Boulder | Yes | A mandatory rule of the guides is broken: code in the wrong layer, an invariant violated, missing or implementation-derived tests, a weakened check, a clear performance regression, the template, the labels or Conventional Commits not followed, undocumented public code. |
| Pebble | No, needs a follow-up issue | An untested edge case, incomplete documentation, a choice left undocumented. |
| Sand | No, to be considered | An alternative design or approach. |
| Dust | No, optional | A minor detail. |

### Findings and action items

Every finding has **exactly one action item**: the step the author can take to resolve
it. A finding without an action item is not reported. Findings are listed from Mountain
to Dust, each in this format:

```text
<label> (<severity>): <what is wrong>, with the rule or the concrete failure.
<path>:<line>   (or: commit <sha> | branch <name> | pull request title | pull request description)
Action: <one imperative, concrete step>
Done when: <a condition that can be verified, e.g. a test that passes>
```

- `<label>` is a [Conventional Comments](https://conventionalcomments.org/) label: `issue`,
  `suggestion`, `question`, `nitpick`, ... When a decision is needed before anything can
  be done, the finding is a `question` and its action is to answer it.
- What happens to an action depends on the level: Mountain and Boulder actions are done in
  the pull request, a Pebble action opens a follow-up issue, a Sand action is answered in
  the review, a Dust action is optional.

### Self-review

Before opening a pull request, run the self-review skill in
[`.agents/skills/review`](.agents/skills/review/SKILL.md) with your own agent (ask it to
"run the review skill", or, if it does not load skills by itself, to "follow
`.agents/skills/review/SKILL.md`"), or go through its checklists yourself if you do not
use one. It
checks the change against the guides, with checks specific to Rust and to pandas,
validates every finding and returns the confirmed ones with their action items. The
result is for the author: it is not a CI check, and the skill never posts it. An author
who posts it on the pull request publishes it, which uses one of the three rounds
above.

## Commands

```bash
uv sync                                    # build the Rust extension and install everything
cargo test -p recsyslearn-core --locked    # core unit tests (fast loop)
uv run pytest                              # Python tests; also runs the quickstart and examples/
```

The checks that must pass before a pull request is opened are listed in one place only,
[docs/development/testing.md › Commands](docs/development/testing.md#commands): it is what
CI runs. Run `cargo clippy` through `uv run` so that the bindings compile against the
project's Python.

## Definition of done

- Tests written first, and passing, at every layer the testing guide requires.
- Docstring with formula and references; `docs/methodology.rst`, the API reference and
  `python/recsyslearn/_core.pyi` updated when they apply.
- Every check of [testing.md › Commands](docs/development/testing.md#commands) passes.
- Commit messages and pull request title follow Conventional Commits; the pull request
  uses the template with its checklist completed.
- The self-review has no Mountain or Boulder finding left open.
