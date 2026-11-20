# Contributing to recsyslearn

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all those involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Share it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Adding or changing a metric or a segmentation](#adding-or-changing-a-metric-or-a-segmentation)
  - [Improving The Documentation](#improving-the-documentation)
- [Development Guidelines](#development-guidelines)


## Code of Conduct

This project and everyone participating agrees with the
[recsyslearn Code of Conduct](https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to <giuliowaitforitdavide@duck.com>.


## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://recsyslearn.readthedocs.io).

Before you ask a question, it is best to search for existing [Issues](https://github.com/giuliowaitforitdavide/recsyslearn/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/giuliowaitforitdavide/recsyslearn/issues/new/choose) with the **Question** template.
- Provide as much context as you can about what you're running into (e.g., error type, expected output, actual output, ...).
- Provide the versions of recsyslearn, Python, pandas and of your operating system.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice 
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project [license](https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/LICENSE).

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://recsyslearn.readthedocs.io). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/giuliowaitforitdavide/recsyslearn/issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Report them privately as described in [SECURITY.md](./SECURITY.md).

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/giuliowaitforitdavide/recsyslearn/issues/new/choose) with the **Bug report** template and fill in every section. Blank issues are disabled.
- Explain the behavior you would expect and the actual behavior. For a wrong value, say how you obtained the value you expected (the formula worked out by hand, a paper, another library).
- Provide a *minimal reproducible example*: the smallest script, with its data built inline, that shows the problem.
- Provide the information you collected in the previous section.

Once it's filed:

- A maintainer will try to reproduce the issue with your example. If it cannot be reproduced, you will be asked for more details.
- If it is reproduced, the issue is confirmed as a `bug` and left to be [implemented by someone](#your-first-code-contribution), starting from a test that reproduces it.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for recsyslearn, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://recsyslearn.readthedocs.io) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/giuliowaitforitdavide/recsyslearn/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/giuliowaitforitdavide/recsyslearn/issues/new/choose). Choose the template that fits and fill in every section:

- **Metric or segmentation**, for a new metric or segmentation or a change to the definition of an existing one. recsyslearn only implements what can be traced to a published definition, so the template asks for the source, the formula, and a small example worked out by hand.
- **Feature request**, for everything else: API, performance, packaging, documentation.

In both cases:

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
- **Explain why this enhancement would be useful** to most recsyslearn users.


### Your First Code Contribution

recsyslearn is a Python package backed by a Rust core. You need [uv](https://docs.astral.sh/uv/) and [rustup](https://rustup.rs); the Rust toolchain is pinned in [rust-toolchain.toml](./rust-toolchain.toml) and rustup installs it the first time you run `cargo`.

1. Pick or open an issue (with a template), then fork the repository and clone your fork
  ```bash
    $ git clone git@github.com:your_name_here/recsyslearn.git
  ```

2. Create the environment, which compiles the Rust core and installs the package with the `dev` dependency group, and install the git hooks
  ```bash
    $ uv sync
    $ uv run pre-commit install
  ```

3. Create a branch named after the type of change and what it does
  ```bash
    $ git switch -c fix/name-of-your-fix   # or feat/..., docs/..., ci/...
  ```

4. Write a failing test first, then the change. After editing Rust code, `uv sync` (or any `uv run`) rebuilds the extension.

5. Check your changes with the commands listed in the [testing guide](./docs/development/testing.md#commands): they are the checks of CI. The quick loop while you work is
  ```bash
    $ cargo test -p recsyslearn-core --locked
    $ uv run pytest
  ```
  `uv run` makes the bindings compile against the Python of the project environment (they need Python 3.11 or later, which may not be your system Python).

6. Run the [self-review](./.agents/skills/review/SKILL.md) on your branch: ask your coding agent to run the review skill or, if you do not use one, go through its checklists yourself. Resolve every Mountain and Boulder finding.

7. Commit with a [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) message, push, and open a pull request with the [pull request template](./.github/pull_request_template.md), filling in its checklist. Pull requests are squash-merged, so the title must be a Conventional Commits message too.

The details of each step are in the [development guidelines](#development-guidelines).

### Adding or changing a metric or a segmentation

recsyslearn only contains metrics that can be traced back to a published definition, and it is developed test-first. Please follow these steps, in this order:

1. **Find the source.** Take the formula (or the procedure) from the paper that introduced it, or from the one that states it most precisely, and read it in the paper itself rather than in another library. Note the equation number. If several variants exist, pick one deliberately and say why.
2. **Write down what the source does not say.** Ties, rounding, empty lists, divisions by zero, the base of logarithms: decide, and prefer the choice that makes the result deterministic and independent of the order of the rows. When in doubt, raise an error instead of guessing.
3. **Write the tests from the formula**, before the implementation:
    - small examples whose expected value is worked out by hand, with the arithmetic visible in the test;
    - the edge cases of step 2;
    - a naive, loop-based transcription of the formula in `tests/reference.py`, compared with the library on the random data of `tests/conftest.py`. It must not share code with the implementation.
4. **Implement** it in `crates/recsyslearn-core` (with Rust unit tests on integer codes), expose it in `crates/recsyslearn-py`, and wrap it in `python/recsyslearn`.
5. **Reproduce a published result, if one exists.** When a paper publishes the data behind a number it prints (recommendation lists, splits, a deterministic baseline), add a test to `tests/integration` that starts from those files and checks the printed number. Download the files from a pinned commit with `paper_data.fetch`, do not commit them, and never adjust anything to make the numbers agree: if the result cannot be reproduced, say so and why in `docs/methodology.rst`.
6. **Document** it: the docstring states the formula, the meaning of the result, the choices of step 2 and the references (with DOI or URL); add the same to `docs/methodology.rst`, and the function to the type stubs in `python/recsyslearn/_core.pyi` if you added a binding.

### Improving The Documentation

Improvements to the documentation are always welcome. The user documentation lives in the [docs](./docs) folder and in the docstrings of [python/recsyslearn](./python/recsyslearn); it is written in [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html), built with [Sphinx](https://www.sphinx-doc.org/en/master/) and published on [Read the Docs](https://recsyslearn.readthedocs.io).
To build it locally, run the last command of the [testing guide](./docs/development/testing.md#commands).
Then open `docs/_build/html/index.html` in your browser. The code of the quickstart is executed by the test-suite, so keep it runnable.

## Development Guidelines

The rules of the project are written once, for people and for coding agents alike:

| Guide | Content |
| --- | --- |
| [Workflow](./docs/development/workflow.md) | Issues and their templates, mandatory labels, branches, Conventional Commits, pull requests and their template, releases. |
| [Architecture](./docs/development/architecture.md) | The Rust core, the PyO3 bindings and the Python API; the data contract; the invariants of each layer; documentation in the code; performance. |
| [Testing](./docs/development/testing.md) | Test-driven development, where expected values come from, the layers of tests, the commands. |
| [Security](./docs/development/security.md) | Secure coding rules, dependencies, CI and secrets. |

[AGENTS.md](./AGENTS.md) is the entry point for coding agents (Claude Code, GitHub Copilot and others) and summarises the rules that are never optional. It also defines how reviews are done (severity scale, one action item per finding) and points to the [self-review skill](./.agents/skills/review/SKILL.md), which any contributor can run with their own agent before opening a pull request.

## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
