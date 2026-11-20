# Workflow

How a change travels from an idea to a release. These rules apply to everyone who
contributes, people and coding agents alike.

## 1. Start from an issue

Every change that is not trivial (a typo, a broken link) starts from an issue, so that the
problem is agreed on before the solution is written.

- Search the [existing issues](https://github.com/giuliowaitforitdavide/recsyslearn/issues) first.
- Open the issue **with one of the templates** in
  [`.github/ISSUE_TEMPLATE`](../../.github/ISSUE_TEMPLATE); blank issues are disabled.

  | Template | Use it for |
  | --- | --- |
  | `bug_report.md` | A result that is wrong, an unexpected error, a crash. |
  | `metric_request.md` | A new metric or segmentation, or a change to the definition of an existing one. It asks for the published source. |
  | `feature_request.md` | Any other change: API, performance, packaging, documentation. |
  | `question.md` | A question on how to use the library or how to read a result. |

- Fill in every section of the template. Write "Not applicable" where a section does not
  apply instead of deleting it.
- Every issue has at least one [label](#labels) from the moment it is opened. The
  templates set it in the browser; from the command line pass both the template and the
  label: `gh issue create --template "Bug report" --label bug`.
- Security vulnerabilities are **not** reported in issues: see [SECURITY.md](../../SECURITY.md).

## 2. Branch

- Never commit to `main`. Branch from an up-to-date `main`.
- Name the branch `<type>/<short-description>` in kebab-case, with the same types as the
  commits: `feat/generalized-cross-entropy`, `fix/kl-zero-target`, `docs/quickstart`.
- One branch, one logical change. Unrelated fixes found along the way get their own issue.

## 3. Develop

Follow the other guides while writing the change:

- [architecture.md](architecture.md): where code goes and which invariants hold;
- [testing.md](testing.md): tests come first;
- [security.md](security.md): what must never be done.

## 4. Commit

Commit messages follow [Conventional Commits 1.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>(<optional scope>)<!>: <description>

<optional body: what changed and why, not how>

<optional footers, e.g. "Closes #123" or "BREAKING CHANGE: ...">
```

- **Types**: `feat` (new behaviour for users), `fix` (bug fix), `perf`, `refactor` (no change
  in behaviour), `docs`, `test`, `build` (packaging, dependencies), `ci`, `chore`
  (everything else, e.g. `chore(release): 4.0.1`).
- **Scopes** (optional) name the area: `accuracy`, `beyond-accuracy`, `fairness`,
  `segmentation`, `core`, `bindings`, `deps`, `release`.
- **Description**: imperative mood, lower case, no final full stop, at most 72 characters:
  `fix(fairness): reject targets with negative probabilities`.
- **Breaking changes** are marked with `!` after the type or scope *and* explained in a
  `BREAKING CHANGE:` footer. A change is breaking when it alters a public signature, the
  columns of an input or output table, or the value a function returns for the same input.
- Commit the lock files (`uv.lock`, `Cargo.lock`) together with the manifest change that
  caused them to change, and nothing generated (`target/`, `.venv/`, `.integration-data/`).

## 5. Check locally

Run what CI runs before pushing (the list is in [testing.md](testing.md#commands)). A pull
request whose checks fail is not ready for review.

Then review your own change with the
[self-review skill](../../.agents/skills/review/SKILL.md) and your coding agent: it checks
the diff against these guides and returns a list of findings, each with one action item.
Resolve the Mountain and Boulder findings before opening the pull request. The scale and
the format are in [AGENTS.md](../../AGENTS.md#review).

## 6. Open the pull request

- Open it against `main` and **use the template**
  [`.github/pull_request_template.md`](../../.github/pull_request_template.md). GitHub
  loads it automatically in the browser; from the command line pass it explicitly and
  then fill it in:

  ```bash
  gh pr create --title "fix(fairness): reject targets with negative probabilities" --body-file .github/pull_request_template.md
  ```

- Keep every section and every checklist item of the template. Tick the items that are
  done; for the ones that do not apply, write why next to them instead of deleting them.
- Give the pull request its [labels](#labels) when you open it (`--label bug`).
- The **title** follows Conventional Commits: pull requests are squash-merged and the
  title becomes the commit message on `main`.
- Link the issue with `Closes #<number>` in the description.
- Keep the pull request small and focused; open it as a draft while it is in progress.

## 7. Review and merge

- All CI checks must pass and a maintainer must approve. Reviews by agents are advisory.
- A pull request gets at most three published rounds of agent review: the first on the
  whole diff, the second and the third only on the answers to the findings of the round
  before. After the third it needs a human reviewer
  ([AGENTS.md](../../AGENTS.md#review)).
- Reviewers follow the rules in [AGENTS.md](../../AGENTS.md#review): every finding cites a
  rule or a concrete failure, has a severity from Mountain to Dust, and one action item.
  The author answers every Mountain and Boulder finding.
- Address review comments with new commits (no force-push after the review has started,
  so that reviewers can see what changed).
- Maintainers squash-merge. The squashed commit keeps the pull request title and number,
  e.g. `fix(fairness): reject targets with negative probabilities (#72)`.

## 8. Release (maintainers)

- The version of the package is defined only in the `[workspace.package]` table of
  [`Cargo.toml`](../../Cargo.toml); the Python package reads it through maturin. The
  `current_version` in the `[tool.bumpversion]` table of `pyproject.toml` is the
  bookkeeping of the release tool, which updates it by itself: never edit either by hand.
- recsyslearn follows [Semantic Versioning](https://semver.org): `fix` and `perf` commits
  go in a patch release, `feat` in a minor one, breaking changes in a major one.
- A release is a pull request like any other:
  1. on a branch `chore/release-X.Y.Z`, run `uv run bump-my-version bump <major|minor|patch>`:
     it updates `Cargo.toml`, `Cargo.lock` and its own `current_version`, and commits
     `chore(release): X.Y.Z`. It creates no tag;
  2. open the pull request with the title `chore(release): X.Y.Z` and squash-merge it
     once CI passes;
  3. publish a GitHub release that creates the tag `vX.Y.Z` on the merged commit of
     `main`. The tag is made here and not by the bump, because a tag made on the release
     branch would not be on the commit that squash-merging puts on `main`.
- Publishing the release runs [`release.yml`](../../.github/workflows/release.yml), which
  builds the wheels and the source distribution from that tag and uploads them to PyPI.

## Labels

Labels are mandatory on every issue and every pull request, from the moment it is opened.

- Use only the labels that already exist in the repository (`gh label list`). Never create
  a new one: propose it in an issue instead.
- An issue takes the label of its template (`bug`, `enhancement` or `question`), plus any
  other existing label that applies (`documentation`, `good first issue`, ...).
- A pull request takes the labels of the issue it closes. When there is no linked issue,
  or the issue has no label, choose by the type of the title:

  | Type of the pull request | Label |
  | --- | --- |
  | `fix` | `bug` |
  | `feat`, `perf`, `refactor`, `test`, `build`, `chore` | `enhancement` |
  | `docs` | `documentation` |
  | `build(deps)` | `dependencies` |
  | `ci` | `github_actions` |

- The command line and the API do not apply labels by themselves: agents pass them
  explicitly with `--label`.

## Dependencies

- Python: `uv add <package>` / `uv lock --upgrade-package <package>`. Rust: `cargo add` /
  `cargo update -p <crate>`. Never edit a lock file by hand.
- Dependabot opens the routine update pull requests (`build` and `ci` prefixes).
- A new dependency, of the package, of the core or of the development tools, needs a
  justification in the pull request: see [security.md](security.md#dependencies).
