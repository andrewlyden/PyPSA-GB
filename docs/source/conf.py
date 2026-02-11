# Configuration file for the Sphinx documentation builder.
#
# Full documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add scripts directory for autodoc
sys.path.insert(0, os.path.abspath('../../../scripts'))
sys.path.insert(0, os.path.abspath('../../../config'))


# -- Project information -----------------------------------------------------

project = 'PyPSA-GB'
copyright = '2021-2026, Andrew Lyden, University of Edinburgh'
author = 'Andrew Lyden'
version = '2.0'
release = '2.0.0'


# -- General configuration ---------------------------------------------------

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # Google/NumPy docstring support
    
    # Markdown support
    "myst_parser",
    
    # Jupyter notebooks
    "nbsphinx",
    "nbsphinx_link",
    
    # UI enhancements
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# MyST-Parser configuration (for .md files)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "fieldlist",
    "attrs_inline",
]
myst_heading_anchors = 3

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}
autosummary_generate = True

# Napoleon settings (for docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pypsa': ('https://pypsa.readthedocs.io/en/stable/', None),
}

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 300

# Configure MIME type rendering priorities - prefer HTML over Plotly JSON
nbsphinx_output_priority = {
    'text/html': 0,
    'application/vnd.plotly.v1+json': -10,  # Deprioritize Plotly JSON
    'application/javascript': 1,
    'text/latex': 2,
    'image/svg+xml': 3,
    'image/png': 4,
    'image/jpeg': 5,
    'text/markdown': 6,
    'text/plain': 7,
}

# Add prolog to handle Plotly and hide code cells by default
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Hide code cells by default, show on click */
        div.nbinput.container {
            display: none;
        }
        div.nbinput.container.show {
            display: block;
        }
        /* Add toggle button styling */
        .toggle-button {
            background-color: #7ba591;
            color: white;
            padding: 6px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin: 10px 0;
            font-size: 0.9em;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(123, 165, 145, 0.2);
        }
        .toggle-button:hover {
            background-color: #93c4ab;
            box-shadow: 0 3px 6px rgba(123, 165, 145, 0.3);
            transform: translateY(-1px);
        }
        /* Hide "Data type cannot be displayed" messages */
        .nboutput p:has-text("Data type cannot be displayed") {
            display: none;
        }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Add toggle buttons for code cells
            var codeCells = document.querySelectorAll('div.nbinput.container');
            codeCells.forEach(function(cell, index) {
                var button = document.createElement('button');
                button.className = 'toggle-button';
                button.textContent = 'Show Code';
                button.onclick = function() {
                    if (cell.classList.contains('show')) {
                        cell.classList.remove('show');
                        button.textContent = 'Show Code';
                    } else {
                        cell.classList.add('show');
                        button.textContent = 'Hide Code';
                    }
                };
                cell.parentNode.insertBefore(button, cell);
            });
            
            // Hide "Data type cannot be displayed" messages
            var outputs = document.querySelectorAll('.nboutput p');
            outputs.forEach(function(p) {
                if (p.textContent.includes('Data type cannot be displayed')) {
                    p.style.display = 'none';
                }
            });
        });
    </script>
"""

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Suppress warnings
suppress_warnings = ['myst.header', 'autosummary']


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "PyPSA-GB",
    },
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "show_nav_level": 1,  # Show current section + children in sidebar
    "navigation_depth": 2,  # Expand 2 levels in sidebar
    "header_links_before_dropdown": 4,  # Move to dropdown after 4 items
    "footer_center": ["copyright"],
    "secondary_sidebar_items": ["page-toc"],
    "use_edit_page_button": True,
    # "announcement": "PyPSA-GB: Great Britain Power System Model",
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# Gurubase "Ask AI" widget
html_js_files = ['gurubase-widget.js']

# Use pydata theme's built-in sidebar navigation (no override needed)

# Logo and favicon (if you have them)
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'
