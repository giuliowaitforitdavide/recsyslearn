---
name: Bug report
about: A wrong result, an unexpected error or a crash
title: ''
labels: bug
assignees: ''

---

<!-- Fill in every section; write "Not applicable" instead of deleting one.
     Security problems are reported privately: see SECURITY.md. -->

## What happened

<!-- What did you observe? Paste the full traceback if there is one. -->

## What you expected

<!-- For a wrong value: the value you expected and how you obtained it (the formula
     worked out by hand, a paper with its equation number, another library). -->

## Minimal reproducible example

<!-- The smallest script that shows the problem, with the data built inline. -->

```python
import pandas as pd
import recsyslearn

recommendations = pd.DataFrame({"user": [...], "item": [...], "rank": [...]})
```

## Environment

- recsyslearn version: <!-- python -c "import recsyslearn; print(recsyslearn.__version__)" -->
- Python version:
- pandas and numpy versions:
- Operating system and architecture: <!-- e.g. Ubuntu 24.04 x86-64, macOS 15 arm64 -->
- Installed from: <!-- PyPI wheel / source distribution / git -->

## Additional context

<!-- Does it also happen with an older version? Anything else that helps. -->
