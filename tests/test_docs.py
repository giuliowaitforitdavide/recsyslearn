"""The code shown in the documentation runs as written."""

from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples" / "__assets__"


def python_blocks(path: Path) -> list[str]:
    """The content of the ``.. code-block:: python`` directives of an RST file."""
    pattern = re.compile(r"^\.\. code-block:: python\n\n((?:(?: {4}.*)?\n)+)", re.MULTILINE)
    return [textwrap.dedent(block) for block in pattern.findall(path.read_text())]


def test_quickstart_runs(monkeypatch):
    blocks = python_blocks(ROOT / "docs" / "user" / "quickstart.rst")
    assert len(blocks) >= 5
    monkeypatch.chdir(ROOT)  # the quickstart reads the files of examples/__assets__
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, "quickstart.rst", "exec"), namespace)
    assert isinstance(namespace["item_groups"], pd.DataFrame)


def test_readme_example_runs(tmp_path, monkeypatch):
    (block,) = python_blocks(ROOT / "README.rst")
    for name, source in {
        "train.csv": "train_dataset.csv",
        "test.csv": "test_dataset.csv",
        "top_k.csv": "top_k.csv",
    }.items():
        shutil.copy(ASSETS / source, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    exec(compile(block, "README.rst", "exec"), {})


@pytest.mark.parametrize("example", sorted((ROOT / "examples").glob("*.py")), ids=lambda p: p.name)
def test_examples_run(example, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    exec(compile(example.read_text(), str(example), "exec"), {"__name__": "__main__"})
    assert capsys.readouterr().out
