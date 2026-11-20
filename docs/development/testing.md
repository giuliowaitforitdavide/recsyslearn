# Testing

recsyslearn is developed test-first. For a library whose only job is to return the right
number, the tests are the specification: they are written from the published definition
before the implementation exists, never derived from what the implementation returns.

## The cycle

Test-driven development, with one extra step in front:

0. **Source.** Read the definition in the paper (see
   [CONTRIBUTING.md](../../CONTRIBUTING.md#adding-or-changing-a-metric-or-a-segmentation))
   and decide every case the paper leaves open.
1. **Red.** Write a test that states the expected behaviour and run it: it must fail, and
   fail for the right reason (the function is missing or returns the wrong value, not a
   typo in the test).
2. **Green.** Write the simplest implementation that makes it pass.
3. **Refactor.** Clean up with the tests green. Then go back to 1 for the next behaviour.

The same cycle applies to bug fixes: first a test that reproduces the bug and fails, then
the fix. A pull request that changes behaviour without a test that would have failed
before it is not complete.

## Where expected values come from

- From the formula, worked out by hand, with the arithmetic visible in the test
  (`expected = 1 / math.log2(3)`, not `expected = 0.6309297535714575`).
- From a number printed in a paper, quoted with table and equation.
- **Never** from running the implementation and pasting its output, and never adjusted
  until the test passes. If the implementation and the source disagree, one of them is
  wrong: find out which.

## Layers

| Layer | Where | What it proves |
| --- | --- | --- |
| Core unit tests | `#[cfg(test)]` modules in `crates/recsyslearn-core/src` | Each formula on integer codes, edge cases, and every error path of the core. |
| API unit tests | `tests/test_<module>.py` | Hand-worked examples through the public API; input validation and error messages; options (`k`, `per_user`, ...). |
| Reference oracle | `tests/reference.py`, used by the `test_agrees_with_the_reference_implementation` tests | A naive, loop-based transcription of each formula agrees with the library on the random data of `tests/conftest.py`. It must not share code with the library. |
| Input handling | `tests/test_inputs.py` | Results do not depend on identifier dtype, index, or the scale of `rank` and `score`. |
| Documentation | `tests/test_docs.py` | The code of the quickstart, of the README and of every script in `examples/` runs as written. |
| Integration | `tests/integration/` | Numbers printed in papers are reproduced from the papers' public data. |

Every public function needs the first three layers. The exception is a function that
computes nothing and therefore has no code in the core, such as
`segmentation.by_feature`, which only relabels a column: its API tests are its
specification. An integration test is added whenever
a paper publishes the data behind a number it prints; when none does, that is recorded in
[`docs/methodology.rst`](../methodology.rst) ("Validation against published results").

## What to test for every function

- At least one example small enough to verify by hand, for each option that changes the
  result.
- The boundaries: empty input, a single user or item, `k` larger than the lists, ties,
  groups that receive nothing, values exactly on a threshold.
- Every documented choice made where the source is silent.
- Every error the function can raise, matching on the message.
- That the inputs are not modified.

## Rules for integration tests

- Start from the files the authors published, downloaded with `paper_data.fetch` from a
  **pinned commit**. Do not commit the data.
- Hard-code the printed values with a reference to the table they come from.
- Any step the paper does not document (tie order, a filter) is an explicit, commented
  assumption, and the test shows how much the result depends on it.
- Never tune anything to make numbers agree. A result that cannot be reproduced is
  reported as such, with the reason.
- They are marked `@pytest.mark.integration` and excluded from the default run because
  they need the network; tests that need no download are left unmarked.

## Conventions

- pytest, plain `assert`, `pytest.approx` for floats (exact `==` only for values that are
  exact in binary floating point), `pytest.raises(..., match=...)` for errors.
- Test names say what is expected (`test_item_crossing_the_threshold_is_in_the_head`),
  not what is called.
- Warnings are errors (`filterwarnings = ["error"]`): a new deprecation fails the suite.
- Tests are deterministic: random data use a fixed seed.

## Commands

```bash
cargo fmt --all --check                                                  # Rust formatting
uv run cargo clippy --workspace --all-targets --locked -- -D warnings    # Rust lints
cargo test -p recsyslearn-core --locked                                  # core unit tests
RUSTDOCFLAGS="-D warnings" cargo doc -p recsyslearn-core --no-deps --locked   # documentation of the core
uv run ruff check .                                                      # Python lints
uv run ruff format --check .                                             # Python formatting
uv run pytest                                                            # Python tests, quickstart and examples
uv run pytest -m integration                                             # reproduction of papers (downloads ~170 MB once)
uv sync --group docs && uv run sphinx-build -W -b html docs docs/_build/html   # documentation
```

This is the only list of the checks: the other guides, the pull request template and
`AGENTS.md` point here. CI differs from it in two ways: its Rust job calls
`cargo clippy` directly, because there the Python of the runner is recent enough for the
bindings, and it also builds the workspace with the minimum supported Rust (the
`rust-version` of `Cargo.toml`), which needs that toolchain installed. All of them must pass before a pull request is opened; the
integration tests may be skipped only when the change touches no metric, no segmentation
and nothing of how the inputs are read.

`uv run` rebuilds the extension when the Rust sources change. While iterating on the core,
`cargo test -p recsyslearn-core` alone is the fastest loop; a single Python test runs with
`uv run pytest tests/test_fairness.py -k entropy`.

CI ([`ci.yml`](../../.github/workflows/ci.yml)) runs all of the above, on every supported
Python version, with the oldest supported numpy and pandas, and on Linux, macOS and Windows.
