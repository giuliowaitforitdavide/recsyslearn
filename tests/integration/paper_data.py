"""Download, once, the public data files the integration tests start from.

Files are fetched from the repositories of the papers at a pinned commit, so that their
content cannot change, and cached under ``.integration-data`` at the root of the
repository (or under ``$RECSYSLEARN_INTEGRATION_DATA``). Nothing is redistributed with
recsyslearn: each file stays under the terms of the repository it comes from.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[2] / ".integration-data"
CACHE = Path(os.environ.get("RECSYSLEARN_INTEGRATION_DATA", _DEFAULT))


def fetch(repository: str, commit: str, path: str) -> Path:
    """Local copy of ``path`` of the GitHub ``repository`` at ``commit``."""
    target = CACHE / repository / commit / path
    if target.exists():
        return target
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "recsyslearn-integration-tests"})
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    partial.replace(target)
    return target
