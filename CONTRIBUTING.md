# Contributing to recsyslearn

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. This guide describes the contribution **process**: how to ask questions, report bugs, propose changes and open pull requests. The **technical rules** (setup, architecture, testing, security, performance, review) are in [`AGENTS.md`](AGENTS.md), which applies to human contributors and AI coding agents alike.

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Mention it on social media
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Legal Notice](#legal-notice)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements and New Metrics](#suggesting-enhancements-and-new-metrics)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)
- [Pull Requests and Review](#pull-requests-and-review)
- [Security](#security)


## Code of Conduct

This project and everyone participating agrees with the
[recsyslearn Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to <giuliowaitforitdavide@duck.com>.


## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://recsyslearn.readthedocs.io).

Before you ask a question, it is best to search for existing [Issues](https://github.com/giuliowaitforitdavide/recsyslearn/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue.

If you then still feel the need to ask a question, open an [Issue](https://github.com/giuliowaitforitdavide/recsyslearn/issues/new/choose) and provide as much context as you can: what you are trying to do, the code you ran, the expected and the actual output, and the versions of Python, recsyslearn, numpy and pandas.


## I Want To Contribute

### Legal Notice

When contributing to this project, you must agree that you have the necessary rights to the content you contribute and that it may be provided under the project [license](LICENSE).

Contributions made with the help of AI coding agents are welcome. The person opening the pull request is responsible for the whole contribution: they must have reviewed and understood it, make sure it does not include content they have no right to contribute, and declare the AI assistance in the pull request template.

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Please complete the following steps in advance:

- Make sure that you are using the latest version.
- Make sure the problem is really a bug and not an error on your side, e.g. incompatible versions of the dependencies or wrong input columns (see the [documentation](https://recsyslearn.readthedocs.io)).
- Check whether a [bug report](https://github.com/giuliowaitforitdavide/recsyslearn/issues?q=label%3Abug) already exists for your problem.
- Collect information about the bug:
  - full traceback;
  - versions of Python, recsyslearn, numpy and pandas, and your OS;
  - a minimal piece of code that reproduces the problem, with its input and output.

#### How Do I Submit a Good Bug Report?

> Never report security vulnerabilities in the issue tracker or elsewhere in public: follow [SECURITY.md](SECURITY.md) instead.

Open an [Issue](https://github.com/giuliowaitforitdavide/recsyslearn/issues/new/choose) with the **Bug report** template and fill in every section. The maintainers will try to reproduce the problem and label the issue accordingly. Bugs that cannot be reproduced will not be addressed until reproduction steps are provided.

### Suggesting Enhancements and New Metrics

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://recsyslearn.readthedocs.io) carefully and find out if the functionality is already covered.
- [Search](https://github.com/giuliowaitforitdavide/recsyslearn/issues) for existing suggestions. If there is one, add a comment to it instead of opening a new issue.
- Find out whether your idea fits with the scope and aims of the project. Keep in mind that we want features that will be useful to the majority of our users.

#### How Do I Submit a Good Enhancement Suggestion?

Open an [Issue](https://github.com/giuliowaitforitdavide/recsyslearn/issues/new/choose) with the matching template:

- **New metric**: to propose a new metric. A metric is accepted only with a verifiable reference (paper, equation, formula), as required by [`AGENTS.md`](AGENTS.md#metrics).
- **Feature request**: for any other enhancement (API, performance, documentation, …).

Features and new metrics start from an approved issue: wait for a maintainer to confirm the proposal before opening a pull request.

### Your First Code Contribution

1. Fork the repository and clone your fork.
2. Set up the environment with the commands in [`AGENTS.md`](AGENTS.md#1-overview-and-commands).
3. Create a branch named as described in [`AGENTS.md`](AGENTS.md#2-workflow), e.g. `fix/ndcg-partial-ats`.
4. Make your changes following [`AGENTS.md`](AGENTS.md), and check that tests and `pre-commit` pass.
5. Push your branch and open a pull request against `main`.

### Improving The Documentation

The documentation is in the [docs](docs) folder, written in [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) and built with [Sphinx](https://www.sphinx-doc.org/en/master/). It is published on [Read the Docs](https://recsyslearn.readthedocs.io) from the `main` branch. The command to build it locally is in [`AGENTS.md`](AGENTS.md#1-overview-and-commands); the output is written to `docs/_build/html`.


## Pull Requests and Review

- Fill in every section of the pull request template, and link the issue the PR closes.
- Use a PR title that follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): PRs are squash-merged and the title becomes the commit message.
- Before opening the PR, you can run a self-review with your own coding agent, as described in [`AGENTS.md`](AGENTS.md#7-review).
- Reviews use the severity scale described in [`AGENTS.md`](AGENTS.md#7-review). Mountain and Boulder findings block the merge until they are fixed or answered.
- Every PR is approved by a human maintainer before merging.


## Security

See [SECURITY.md](SECURITY.md).


## Attribution
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
