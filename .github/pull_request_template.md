<!-- Title: Conventional Commits, e.g. "fix(fairness): reject targets with negative probabilities".
     Pull requests are squash-merged: the title becomes the commit message on main.
     Keep every section and every checklist item. Tick what is done; where an item does not
     apply, leave it unticked and write why next to it. -->

## Summary

<!-- What does this pull request change, and why? -->

Closes #

## Type of change

<!-- Tick one; it must match the type in the title. -->

- [ ] `feat`: new behaviour for users
- [ ] `fix`: bug fix
- [ ] `perf`: faster or lighter, same results
- [ ] `refactor`: no change in behaviour
- [ ] `docs`: documentation only
- [ ] `test`: tests only
- [ ] `build` / `ci` / `chore`: packaging, dependencies, automation
- [ ] Breaking change (`!` in the title and a `BREAKING CHANGE:` note below)

## How it was tested

<!-- Which tests were written first and failed before the change? Which commands did you run? -->

## Checklist

- [ ] The change started from an issue opened with a template (linked above).
- [ ] The issue and this pull request have their labels, chosen among the existing ones ([workflow guide](https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/docs/development/workflow.md#labels)).
- [ ] Tests were written before the implementation and failed without it ([testing guide](https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/docs/development/testing.md)).
- [ ] Every check listed in the [testing guide › Commands](https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/docs/development/testing.md#commands) passes locally.
- [ ] The layers and their invariants are respected, code is documented with docstrings (inline comments only for particular logic), and hot paths are not slower ([architecture guide](https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/docs/development/architecture.md)).
- [ ] No new dependency, secret, generated file or weakened check ([security guide](https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/docs/development/security.md)).
- [ ] The self-review skill (`.agents/skills/review`) was run, or the change was self-reviewed by hand, and no Mountain or Boulder finding is left open.
- [ ] Documentation is updated: docstrings, `docs/`, `README.rst` and `python/recsyslearn/_core.pyi` where they apply.

### For a new or changed metric / segmentation

- [ ] The definition comes from a published source, cited with DOI or URL and equation number in the docstring and in `docs/methodology.rst`.
- [ ] Expected values in the tests come from the formula worked out by hand, and the library is compared with a naive transcription of the formula in `tests/reference.py`.
- [ ] Every choice the source does not specify (ties, rounding, undefined cases) is tested, documented in the docstring and listed in `docs/methodology.rst`.
- [ ] A published result is reproduced in `tests/integration`, or `docs/methodology.rst` says why none can be.

## Notes for the reviewer

<!-- Anything that deserves attention: trade-offs, follow-ups, a BREAKING CHANGE description. -->
