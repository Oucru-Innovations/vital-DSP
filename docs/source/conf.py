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
copyright = '2024, VitalDSP Team'
author = 'van-koha'
release = '0.1.4'
version = '0.1.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinx_markdown_builder',
    "myst_nb",
    "jupyter_sphinx",          # for widget-backed outputs (fallback) # registers plotly MIME + JS
    "sphinxcontrib.jquery",    # ensures jQuery is available to extensions that expect it
]

# Execute notebooks during build (recommended for RTD)
nb_execution_mode = "auto"           # "auto" or "force"
nb_execution_timeout = 180           # bump if  notebooks are heavy
nb_render_plugin = "default"         # myst-nb’s default HTML render

# MyST config (optional, but useful)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
]

# HTML/theme
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- sphinxcontrib-plotly options -------------------------------------------
# Use RTD-hosted bundled JS (default). Set to True to inline plotly.js if needed.
plotly_include_plotlyjs = True

# -- Plotly configuration for ReadTheDocs -----------------------------------
# Plotly is already configured above with plotly_include_plotlyjs = True

# Additional MyST configurations (optional)
myst_enable_extensions = [
    "dollarmath",  # Use $...$ syntax for math
    "amsmath",     # Use LaTeX math environments
    "deflist",     # Use definition lists
    "html_admonition",  # Use admonition with HTML support
    "html_image",  # Use HTML-like <img> tags for images
    "colon_fence", # Use ::: for extended Markdown syntax
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': False,
    'inherited-members': False,
    'show-inheritance': True,
    'exclude-members': '__weakref__,__dict__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'plotly': ('https://plotly.com/python/', None),
    'dash': ('https://dash.plotly.com/', None),
}

# # Exclude notebooks from autosummary processing
# autosummary_generate = [
#     'advanced_computation.rst',
#     'filtering.rst',
#     'index.rst',
#     # Exclude the ipynb files from autosummary
#     # 'notebooks/signal_filtering.ipynb',
#     # 'notebooks/transforms.ipynb',
#     'respiratory_analysis.rst',
#     'signal_quality_assessment.rst',
#     'time_domain.rst',
#     'transforms.rst',
#     'utils.rst'
# ]


source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

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
    'style_external_links': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'style_nav_header_background': '#2980B9',  # Custom color for the header
    'canonical_url': 'https://vital-dsp.readthedocs.io/',
    'analytics_id': '',  # Provided by Google Analytics
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
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

# nbsphinx configuration
nbsphinx_allow_errors = True  # Allow notebook execution errors
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_timeout = 60  # Timeout for notebook execution

# Plotly configuration for nbsphinx
nbsphinx_prolog = """
.. raw:: html

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
"""

# Suppress warnings
suppress_warnings = ['nbsphinx']