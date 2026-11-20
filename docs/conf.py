"""Sphinx configuration of the recsyslearn documentation."""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Document the sources directly: the native module is mocked, so building the
# documentation needs neither a Rust toolchain nor an installed package.
sys.path.insert(0, str(ROOT / "python"))
autodoc_mock_imports = ["recsyslearn._core"]

project = "Recsyslearn"
copyright = "2022, Giulio Davide Carparelli"
author = "Giulio Davide Carparelli"
# The single source of truth for the version is the Cargo workspace.
version = release = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"][
    "version"
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_use_rtype = False

templates_path = ["_templates"]
# docs/development holds the contributor guides, which are read on GitHub.
exclude_patterns = ["_build", "development", "Thumbs.db", ".DS_Store", "**/site-packages/*"]
pygments_style = "sphinx"

# The theme of NumPy, pandas, SciPy and scikit-learn: readers of a scientific library
# already know their way around it.
html_theme = "pydata_sphinx_theme"
html_title = "recsyslearn"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
        "text": "recsyslearn",
        "alt_text": "recsyslearn",
    },
    "github_url": "https://github.com/giuliowaitforitdavide/recsyslearn",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/recsyslearn/",
            "icon": "fa-brands fa-python",
        }
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
# Follow the light / dark preference of the reader's system.
html_context = {"default_mode": "auto"}
# The landing page has no section navigation to show on the left.
html_sidebars = {"index": []}
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_css_files = ["custom.css"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}
