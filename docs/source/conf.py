# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import sphinx_rtd_theme

# Add the source directory to sys.path
sys.path.insert(0, os.path.abspath('../../src'))

project = 'VitalDSP'
copyright = '2024, van-koha'
author = 'van-koha'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx_rtd_theme', 
    'sphinx_markdown_builder',
    'm2r2'
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the theme to 'sphinx_rtd_theme' for a modern look
html_theme = 'sphinx_rtd_theme'

# Add custom static files (e.g., custom CSS) to enhance the appearance
html_static_path = ['_static']

# Additional options for the Read the Docs theme
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'style_nav_header_background': '#2980B9',  # Custom color for the header
}

# Custom sidebar templates, must be a dictionary that maps document names to template names.
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',  # needs 'show_related': True theme option to display
        'sourcelink.html',
        'searchbox.html',
    ]
}

# -- Options for Extensions -------------------------------------------------
todo_include_todos = True  # Enable the todo extension if needed
