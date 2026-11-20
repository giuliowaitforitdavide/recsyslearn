# Architecture

recsyslearn is a Python package whose computations run in a Rust core. This guide says
what lives where, how data flows, and which invariants every change must preserve.

## Layers

```text
pandas DataFrames                                            floats / Series / DataFrames
        │                                                                 ▲
        ▼                                                                 │
┌──────────────────────────────────────────────────────────────────────────────────┐
│ python/recsyslearn        public API: validate tables, map identifiers to dense  │
│                           integer codes, call the core, label the results        │
├──────────────────────────────────────────────────────────────────────────────────┤
│ crates/recsyslearn-py     PyO3 bindings: borrow NumPy arrays, release the GIL,   │
│ (module recsyslearn._core) convert errors. No logic                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│ crates/recsyslearn-core   pure Rust: every formula, on integer codes             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

| Path | Responsibility |
| --- | --- |
| `crates/recsyslearn-core/src/lists.rs` | `RankedLists` and `GroundTruth`: the long tables regrouped by user (compressed sparse rows). |
| `crates/recsyslearn-core/src/{accuracy,beyond_accuracy,fairness,segmentation}.rs` | One module per family of functions, each with its formulas in the module documentation and its unit tests. |
| `crates/recsyslearn-core/src/error.rs` | The `Error` type and the shared input checks. |
| `crates/recsyslearn-py/src/lib.rs` | The `recsyslearn._core` extension module. |
| `python/recsyslearn/_frames.py` | Shared validation of DataFrames and conversion of identifiers to codes. |
| `python/recsyslearn/{accuracy,beyond_accuracy,fairness,segmentation}.py` | The public functions and their docstrings (formula, parameters, references). |
| `python/recsyslearn/errors.py` | The public exceptions. |
| `python/recsyslearn/_core.pyi` | Type stubs of the extension module. |
| `tests/` | Python tests; see [testing.md](testing.md). |
| `docs/` | Sphinx documentation (`docs/development` holds these guides and is not part of the built site). |
| `.agents/skills/` | Procedures for coding agents in the [Agent Skills](https://agentskills.io) format (`.claude/skills` links to them for Claude Code). |

## Data flow of a call

1. **Validate** (Python): the argument is a DataFrame with the required columns
   (`MissingColumnsError`), options are valid (`InvalidInputError`).
2. **Encode** (Python): identifiers of any hashable type become dense `uint32` codes with
   `pandas.factorize`. Tables that must agree (recommendations and ground truth) are
   encoded together so that they share the codes.
3. **Compute** (Rust): contiguous NumPy arrays are borrowed without copying, the GIL is
   released, and the core returns plain numbers or arrays indexed by code. Per-user work
   is parallelised with rayon.
4. **Label** (Python): arrays are turned back into floats, or Series / DataFrames indexed
   by the original identifiers.

## The data contract

- Input tables are long-format DataFrames with the columns `user` and `item`. In
  recommendations each user's list is ordered by `rank` (1 = best), else by `score`
  (highest = best), else by the order of the rows. In interactions `rating` is the
  explicit feedback value, used as graded relevance when the table is a ground truth.
  The names are those most tools use.
- Group tables have the columns `user` or `item`, and `group`.
- Formulas use the *position* of an item in the sorted list, never the raw `rank` or
  `score`. Two items of a user cannot have the same `rank`; they can have the same
  `score`, and then no order is chosen for them: a quantity that is a sum over positions
  (a discounted gain, an exposure, the self-information of novelty) is replaced by its
  expected value over the orders of the tie, and what a metric derives from it is
  computed on that expected value; a metric that is not such a sum (coverage) refuses a
  tie that the cutoff splits. Which of these has a published source is recorded in
  [`docs/methodology.rst`](../methodology.rst). Tables that must agree cannot hold identifiers of different types.
- The full contract is documented for users in [`docs/user/data.rst`](../user/data.rst);
  change both together.

## Invariants

These hold everywhere; a change that needs to break one must be discussed in an issue first.

**Core (`recsyslearn-core`)**

- It does not depend on Python and knows nothing about identifiers, labels or pandas.
- No `unsafe` code (`unsafe_code = "forbid"` in the workspace lints).
- No panic on user input: every public function that computes or builds something
  validates lengths, code bounds and value domains and returns `Result<_, Error>`.
  Accessors that cannot fail are total instead: asked about a user that does not exist,
  they return an empty slice or zero. `unwrap`, `expect`, indexing and arithmetic are
  only allowed where a preceding check makes failure impossible.
- Deterministic: the same input gives the same output, and the same error, whatever the
  number of threads: sums are made in the order of the codes, and of several invalid
  users the one with the lowest code is reported. Ties are broken explicitly (ascending
  code), never by hash order.
- Memory follows the rows and the number of users the caller declares (`n_users`), never
  the largest item code: item codes need not be dense. Every allocation that an argument
  sizes is fallible and returns an error instead of aborting. The limit of that
  guarantee is the operating system: where memory is overcommitted a reservation
  succeeds and the failure comes when the pages are written. The Python package only
  ever declares the number of identifiers it found in the tables, so this concerns
  direct callers of the core.
- No I/O: no files, no network, no environment variables, no logging.
- It passes `clippy::pedantic` with `-D warnings` and builds on the `rust-version` of
  [`Cargo.toml`](../../Cargo.toml).

**Bindings (`recsyslearn-py`)**

- No logic: argument extraction, a call into the core inside `Python::detach`, conversion
  of the result. Core errors become `recsyslearn.errors.InvalidInputError`.
- Every binding has an entry in `python/recsyslearn/_core.pyi`.

**Python package**

- Functions never modify their arguments and have no side effects.
- No per-row Python work: no `DataFrame.apply`, `iterrows` or Python loops over rows. Bulk
  operations go through pandas / NumPy, formulas go to the core. Every term of a
  documented formula is a formula, including the mean over users of a per-user metric
  (`_core.mean_over_users`), a normalisation and the default of an argument that enters
  the formula (the parity target of `kl_divergence`).
- Where the rows of a table are a set, their order does not change a result, bit for bit:
  identifiers are coded in their sort order (and an optional table such as a target is
  aligned to it), so the core always adds things up in the same order. It costs a sort of
  the distinct identifiers per call. The exceptions are the tables whose row order
  carries meaning (recommendations with neither `rank` nor `score`; the output of `by_feature` follows
  its input) and a column whose identifiers cannot be compared with each other, which
  keeps the order of first appearance.
- A table is validated whole, whichever of its rows a metric ends up reading.
- Only `MissingColumnsError`, `InvalidInputError` (both subclasses of `RecsyslearnError`
  and `ValueError`) and `TypeError` are raised deliberately, with a message that says what
  is wrong and in which argument.
- The public API is what the five public modules list in `__all__`. Names starting with
  an underscore (`_core`, `_frames`) are private and can change without notice.
- Every public function documents its formula and cites its source; the same goes in
  [`docs/methodology.rst`](../methodology.rst).
- A public function that computes nothing (`segmentation.by_feature` relabels a column)
  stays entirely in the Python package: there is no formula to move to the core.

## Documentation in the code

- **Python is documented with docstrings, not comments.** Every module, every public
  class and function, and every helper whose purpose is not evident from its name has a
  docstring; the docstring of a public function also states its formula and its source.
  A test is documented by its name and by the docstring of its module or class, and gets
  a docstring of its own when the derivation of the expected value needs explaining.
- **Rust is documented with documentation comments** (`///` on items, `//!` on modules),
  which are Rust's equivalent of docstrings. The `missing_docs` lint requires them on
  every public item; the documentation of a module carries its formulas.
- **Inline comments (`#`, `//`) are for particular logic only**: a numerical trick, a
  workaround for a behaviour of pandas or of the compiler, the reason why an apparently
  simpler approach would be wrong, a step taken from a paper's protocol, the derivation
  of an expected value in a test, the provenance of a constant (table, equation). Such a
  comment says *why* or *where from*, never *what* the next line does.
- No commented-out code.
- Tool directives (`# noqa`, `# type: ignore`, `# pragma: no cover`, `# fmt: skip`,
  `#[allow(...)]`) name the specific rule and carry a short reason, next to them or on
  the line above.
- The scripts in `examples/` are teaching material: there, comments addressed to the
  reader are part of the documentation.

Ruff enforces the mechanical part in Python (missing docstrings in the package,
commented-out code, blanket directives); the judgement on whether a comment is needed is
for the author and the reviewer.

## Performance

recsyslearn is expected to handle tens of millions of rows. Correctness comes first, but
a change must not make a function asymptotically slower or heavier.

- **Python layer**: bulk pandas / NumPy operations only. No `DataFrame.apply`, `iterrows`,
  `itertuples`, list comprehensions or Python loops over rows, users or items. Each table
  is encoded once per call; no merge that can multiply rows, no `concat` inside a loop.
- **Boundary**: arrays are handed to the core contiguous and in their final dtype, so that
  they are borrowed without a copy, and the GIL is released around the computation.
- **Core**: time linear or `n log n` in the number of rows and memory linear in it. Work
  on arrays indexed by code (counting sorts, compressed sparse rows) rather than hash maps;
  no allocation inside a per-row loop; never a dense user × item structure.
- **Parallelism**: per-user work may run in parallel with rayon, as long as the result
  does not depend on the number of threads (see the determinism invariant): collect
  per-user values in order, do not reduce floating-point sums in parallel.
- **Measure**: `uv run python benchmarks/benchmark.py [n_users] [n_items] [list_length]`
  times every public function on synthetic data. A change to a hot path reports the
  timings before and after in the pull request. A performance claim, in either direction,
  needs numbers.

## Adding a function

1. Core: add it to the relevant module of `recsyslearn-core` (or a new module), operating
   on codes, with its formula in the documentation.
2. Bindings: expose it in `crates/recsyslearn-py/src/lib.rs` and in `_core.pyi`.
3. Python: add the public function with validation, encoding and labelling, and list it
   in `__all__`.
4. Documentation: docstring, `docs/api_ref`, `docs/methodology.rst`, and the user guide if
   the data contract or the workflow changes.

The order in which this is *written* is set by [testing.md](testing.md): tests first.

## Decisions worth knowing

- **Why integer codes instead of passing identifiers to Rust?** `pandas.factorize` is
  already a fast hash pass, and codes let the core use arrays and counting sorts instead
  of hash maps, for any identifier type.
- **Why two crates?** The core can be tested and reused without a Python interpreter, and
  the bindings stay small enough to review at a glance.
- **Why abi3?** One wheel per platform works on every supported Python version.
- **Why functions instead of classes?** The metrics are stateless; a function per metric
  is the smallest API that states its inputs.
