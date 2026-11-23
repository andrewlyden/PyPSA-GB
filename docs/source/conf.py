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
version = '2.3'
release = '2.3.0'


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
    "dollarmath",
    "substitution",
]
myst_heading_anchors = 3

# Each chapter's run figures, generated from the model outputs (the bill chapter
# by scripts/analysis/build_bill_docs_values.py, rule build_bill_docs_values).
# The pages use them as {{ name }}; the tables are {include}d from the chapter's
# _generated folder, wherever it sits under the source tree. Every _generated
# folder must have its values.json, and no name may be defined twice.
import glob as _glob
import json as _json

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_GENERATED_DIRS = sorted(
    os.path.relpath(path, _SOURCE_DIR).replace(os.sep, '/')
    for path in _glob.glob(os.path.join(_SOURCE_DIR, '**', '_generated'), recursive=True)
    if os.path.isdir(path) and '_legacy' not in os.path.relpath(path, _SOURCE_DIR).split(os.sep))
myst_substitutions = {}
_SUBSTITUTION_SOURCE = {}
for _generated in _GENERATED_DIRS:
    _values = os.path.join(_SOURCE_DIR, *_generated.split('/'), 'values.json')
    if not os.path.isfile(_values):
        raise FileNotFoundError(
            f"{_values} is missing: run the chapter's generator (the bill chapter's is "
            "python scripts/analysis/build_bill_docs_values.py, rule build_bill_docs_values)")
    with open(_values, encoding='utf-8') as _f:
        for _key, _value in _json.load(_f)['values'].items():
            if _key in myst_substitutions:
                raise ValueError(f"substitution {_key!r} is defined in both "
                                 f"{_SUBSTITUTION_SOURCE[_key]} and {_values}")
            myst_substitutions[_key] = _value
            _SUBSTITUTION_SOURCE[_key] = _values

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
    'pypsa': ('https://docs.pypsa.org/latest/', None),
}

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_widgets_path = ''  # no Jupyter widgets in the tutorials
# Gallery thumbnails: each tutorial tags one figure cell "nbsphinx-thumbnail".
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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints',
                    '_includes/*',   # fragments that pages include (the home page's contributor list)
                    # Public ownership: kept for the next release, not published in v2.3.0 (user, 18 Nov 2026)
                    'model/after_the_solve/public_ownership/*', 'studies/public_ownership_bills.md']

# Suppress warnings
suppress_warnings = ['myst.header', 'autosummary']


def _exclude_generated(app, config):
    """Generated tables are included into pages, never documents of their own.

    Added after the config is read so a command-line ``-D exclude_patterns=...`` keeps it.
    """
    patterns = [f'{generated}/*' for generated in _GENERATED_DIRS]
    config.exclude_patterns = [*config.exclude_patterns,
                               *(p for p in patterns if p not in config.exclude_patterns)]


def _refresh_contributors(app, config):
    """The home page's code contributors from GitHub (scripts/utilities/build_contributors.py).

    Offline, rate-limited or ``PYPSA_GB_DOCS_OFFLINE`` set: the committed list stays as it is.
    """
    if os.environ.get('PYPSA_GB_DOCS_OFFLINE'):
        return
    try:
        import importlib.util
        script = os.path.join(_SOURCE_DIR, '..', '..', 'scripts', 'utilities', 'build_contributors.py')
        spec = importlib.util.spec_from_file_location('build_contributors', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    except Exception as error:   # keep building with the committed list
        print(f'Contributors not refreshed from GitHub ({error}); using the committed list')


def _repo_stats(app, config):
    """Stars and forks for the sidebar repo card, written in at build time (_templates/sidebars/repo-stats.html).

    Fetched from GitHub when online and cached in _includes/repo_stats.json; offline, rate-limited or
    ``PYPSA_GB_DOCS_OFFLINE`` set, the cached counts are used. The theme still refreshes them in the browser.
    """
    import json
    import urllib.request
    cache = os.path.join(_SOURCE_DIR, '_includes', 'repo_stats.json')
    stats = {}
    if os.path.exists(cache):
        with open(cache, encoding='utf-8') as handle:
            stats = json.load(handle)
    if not os.environ.get('PYPSA_GB_DOCS_OFFLINE'):
        try:
            request = urllib.request.Request('https://api.github.com/repos/andrewlyden/PyPSA-GB',
                                             headers={'User-Agent': 'pypsa-gb-docs'})
            with urllib.request.urlopen(request, timeout=10) as response:
                repo = json.load(response)
            import datetime
            stats = {'repository': repo['full_name'], 'stars': int(repo['stargazers_count']),
                     'forks': int(repo['forks_count']), 'fetched': datetime.date.today().isoformat()}
            with open(cache, 'w', encoding='utf-8') as handle:
                json.dump(stats, handle, indent=2)
                handle.write("\n")
        except Exception as error:   # keep the cached counts
            print(f'Repository stats not refreshed from GitHub ({error}); using the cached counts')
    config.html_context = {**config.html_context, 'repo_stars': stats.get('stars'),
                           'repo_forks': stats.get('forks')}


def setup(app):
    app.connect('config-inited', _exclude_generated)
    app.connect('config-inited', _refresh_contributors)
    app.connect('config-inited', _repo_stats)


# -- Options for HTML output -------------------------------------------------

html_theme = "shibuya"
html_title = "PyPSA-GB"
html_theme_options = {
    "accent_color": "jade",          # overridden by the sage scale in _static/pypsa-gb.css
    "github_url": "https://github.com/andrewlyden/PyPSA-GB",
    "nav_links": [
        {"title": "Getting Started", "url": "getting_started/index"},
        {"title": "Model Description", "url": "model/index"},
        {"title": "Studies", "url": "studies/index"},
        {"title": "Tutorials", "url": "tutorials/index"},
        {"title": "Reference", "url": "reference/index"},
    ],
    "globaltoc_expand_depth": 0,      # sidebar sections closed until clicked (user, 18 Nov 2026)
}
html_context = {
    "source_type": "github",
    "source_user": "andrewlyden",
    "source_repo": "PyPSA-GB",
    "source_version": "master",
    "source_docs_path": "/docs/source/",
}

html_static_path = ['_static']
html_css_files = ['pypsa-gb.css']

# Mermaid diagrams: 16 px text in the site's sage palette, drawn at natural size (_static/pypsa-gb.js).
mermaid_height = "auto"
mermaid_light_theme = "base"
mermaid_dark_theme = "base"
mermaid_init_config = {
    "startOnLoad": False,
    "flowchart": {"useMaxWidth": True, "htmlLabels": True, "nodeSpacing": 40, "rankSpacing": 50,
                  "padding": 14, "curve": "basis"},
    "themeVariables": {
        "fontSize": "16px",
        "primaryColor": "#e5f0ea", "primaryTextColor": "#1f3a2e", "primaryBorderColor": "#7ba591",
        "lineColor": "#5c8b75", "secondaryColor": "#e3eef6", "tertiaryColor": "#fbeee3",
        "clusterBkg": "#f3f8f5", "clusterBorder": "#97c0ac", "edgeLabelBackground": "#ffffff",
    },
}

# Gurubase "Ask AI" widget
html_js_files = ['gurubase-widget.js', 'pypsa-gb.js']


# Browser-tab icon: a small network on the docs' sage green (SVG; every current browser shows it)
html_favicon = '_static/favicon.svg'
