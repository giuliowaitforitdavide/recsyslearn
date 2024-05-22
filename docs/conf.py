project = "Recsyslearn"
copyright = "2022, Giulio Davide Carparelli"
author = "Giulio Davide Carparelli"
version = "2.0.2"


extensions = [
    "sphinx.ext.autodoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/site-packages/*"]
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
