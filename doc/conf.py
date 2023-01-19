# -*- coding: utf-8 -*-
#
# Pastas documentation build configuration file, created by
# sphinx-quickstart on Wed May 11 12:38:06 2016.

# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import re
import sys
from datetime import date
import requests

year = date.today().strftime("%Y")

os.environ["MPLBACKEND"] = "Agg"

from dataclasses import dataclass, field
import sphinxcontrib.bibtex.plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

from pastas.version import __version__

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("."))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    "nbsphinx",
    "numpydoc",
    "sphinx_gallery.load_style",
    "sphinxcontrib.bibtex",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = ".rst"
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Pastas"
copyright = "{}, R.A. Collenteur, M. Bakker, R. Calje, F. Schaars".format(year)
author = "R.A. Collenteur, M. Bakker, R. Calje, F. Schaars"

# The version.
version = __version__
release = __version__
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**groundwater_paper", "**.ipynb_checkpoints"]

add_function_parentheses = False
add_module_names = False
show_authors = False  # section and module author directives will not be shown

pygments_style = "sphinx"  # The name of the Pygments (syntax highlighting) style to use

todo_include_todos = False  # Do not show TODOs in docs

# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_short_title = "Pastas"
html_favicon = "_static/favo.ico"
html_css_files = ["css/custom.css"]
html_show_sphinx = True
html_show_copyright = True
htmlhelp_basename = "Pastasdoc"  # Output file base name for HTML help builder.
html_use_smartypants = True
html_show_sourcelink = True

html_theme_options = {
    "github_url": "https://github.com/pastas/pastas",
    "use_edit_page_button": False,
}

html_context = {
    "github_user": "pastas",
    "github_repo": "pastas",
    "github_version": "master",
    "doc_path": "doc",
}

napoleon_use_param = True
napoleon_type_aliases = {
    "array-like": ":term:`array-like <array_like>`",
    "array_like": ":term:`array_like`",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autosummary_generate = True
numpydoc_class_members_toctree = False


# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# -- Generating references and publications lists -------------------------

# support Round brackets
def bracket_style() -> BracketStyle:
    return BracketStyle(
        left="(",
        right=")",
    )


@dataclass
class MyReferenceStyle(AuthorYearReferenceStyle):
    bracket_parenthetical: BracketStyle = field(default_factory=bracket_style)
    bracket_textual: BracketStyle = field(default_factory=bracket_style)
    bracket_author: BracketStyle = field(default_factory=bracket_style)
    bracket_label: BracketStyle = field(default_factory=bracket_style)
    bracket_year: BracketStyle = field(default_factory=bracket_style)


sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "author_year_round", MyReferenceStyle
)

# Generate bibliography-files from Zotero library
# Get a Bibtex reference file from the Zotero group for referencing
url = "https://api.zotero.org/groups/4846685/collections/8UG7PVLY/items/"
params = {"format": "bibtex", "style": "apa", "limit": 100}

r = requests.get(url=url, params=params)
with open("references.bib", mode="w") as file:
    file.write(r.text)

# Get a Bibtex reference file from the Zotero group for publications list
url = "https://api.zotero.org/groups/4846685/collections/Q4F7R59G/items/"
params = {"format": "bibtex", "style": "apa", "limit": 100}

r = requests.get(url=url, params=params)
with open("publications.bib", mode="w") as file:
    # Replace citation key to prevent duplicate labels and article now shown
    text = re.sub(r"(@([a-z]*){)", r"\1X_", r.text)
    file.write(text)

# Add some settings for bibtex
bibtex_bibfiles = ["references.bib", "publications.bib"]
bibtex_reference_style = "author_year_round"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "Pastas.tex",
        "Pastas Documentation",
        "R.A. Collenteur, M. Bakker, R. Calje, F. Schaars",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = '_static\\logo.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pastas", "Pastas Documentation", [author], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Pastas",
        "Pastas Documentation",
        author,
        "Pastas",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/objects.inv", None),
    "matplotlib": ("https://matplotlib.org/stable/objects.inv", None),
}

# Allow errors in notebooks, so we can see the error online
nbsphinx_allow_errors = True
