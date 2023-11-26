import recsyslearn

project = "Recsyslearn"
copyright = "2022, Giulio Davide Carparelli"
author = "Giulio Davide Carparelli"
version = recsyslearn.__version__


extensions = [
    "sphinx.ext.autodoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
