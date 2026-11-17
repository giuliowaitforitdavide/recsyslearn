---
name: New metric
about: Propose a new metric. A verifiable reference (paper, equation, formula) is required.
title: 'Add <metric name> metric'
labels: 'enhancement'
assignees: ''

---

<!-- Fill in every section. Write N/A if a section does not apply; do not delete it. -->
<!-- See the "Metrics" section of AGENTS.md for the rules every metric must follow. -->

## Metric

- Name:
- Family: <!-- accuracy / beyond-accuracy / fairness -->

## Reference

- Paper: <!-- authors, title, venue, year, DOI or link -->
- Equation number:

## Formula

<!-- The formula as defined in the paper, in LaTeX. -->

```latex

```

## Variant

<!-- If the literature has more than one variant (e.g. different gain or normalisation), which one should be implemented and why. -->

## Per-user or population-level

<!-- Can the metric be computed for each user (returned per user and as the mean over users), or is it defined only over the whole user population (aggregate value only)? -->

## Inputs and edge cases

- Required input columns: <!-- e.g. user, item, rank, group -->
- Empty recommendation lists:
- Users without relevant items:
- Zero denominators:

## Worked example

<!-- A small example (a few users and items) with the expected value computed by hand from the formula. It will become the reference test. -->

## Additional context

<!-- Use cases, other libraries implementing it, anything else. -->
