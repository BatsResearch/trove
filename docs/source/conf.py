project = "Trove"
copyright = "2025, Reza Esfandiarpoor"
author = "Reza Esfandiarpoor"

import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parents[2].joinpath("src").resolve().as_posix())

extensions = [
    "sphinx.ext.duration",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

napoleon_include_init_with_doc = True

## mock dependencies

config_file = Path(__file__).parents[2].joinpath("pyproject.toml")
try:
    import tomllib

    with open(config_file, "rb") as f:
        obj = tomllib.load(f)
except:
    import toml

    with open(config_file, "r") as f:
        obj = toml.load(f)

deps = obj["project"]["dependencies"]
autodoc_mock_imports = [d.split(">=")[0].split("==")[0].strip() for d in deps]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for Autodoc --------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "Trove"
html_static_path = ["_static"]

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/BatsResearch/trove",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],
}

html_favicon = "https://huggingface.co/datasets/BatsResearch/trove-lib-documentation-assets/resolve/main/logo/favicon.svg"
