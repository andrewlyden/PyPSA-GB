# Building PyPSA-GB Documentation

This guide explains how to build and serve the Sphinx documentation locally.

## Prerequisites

The documentation build tools are included in the `pypsa-gb` conda environment:

```bash
conda activate pypsa-gb
```

## Building HTML Documentation

### Quick Build (Linux/Mac)

```bash
cd docs
make html
```

### Quick Build (Windows PowerShell)

```powershell
cd docs
.\make.bat html
```

### Quick Build (Platform-independent Python)

```bash
cd docs
python -m sphinx -b html source build/html
```

This generates HTML files in `docs/build/html/`. Open `docs/build/html/index.html` in your browser.

### Clean and Rebuild (Linux/Mac)

```bash
cd docs
make clean
make html
```

### Clean and Rebuild (Windows PowerShell)

```powershell
cd docs
.\make.bat clean
.\make.bat html
```

### Clean and Rebuild (Platform-independent Python)

```bash
cd docs
# Remove build directory
python -c "import shutil; shutil.rmtree('build/html', ignore_errors=True)"
# Rebuild
python -m sphinx -b html source build/html
```

### Using a Live-Reload Server (Optional)

Install `sphinx-autobuild` for automatic rebuilding on file changes:

```bash
pip install sphinx-autobuild
```

Then from the `docs/` directory:

```bash
sphinx-autobuild source build/html
```

This automatically opens a browser at `http://localhost:8000` and refreshes when you save changes.

### Using Python's HTTP Server (Windows PowerShell)

After building with `make.bat` or Python, serve locally:

```powershell
cd docs\build\html
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser. Press `Ctrl+C` to stop the server.

## Documentation Structure

```
docs/
├── source/                          # Documentation source files
│   ├── conf.py                      # Sphinx configuration
│   ├── index.rst                    # Main landing page
│   ├── getting_started/             # Installation & quickstart
│   ├── user_guide/                  # Usage documentation
│   ├── data_reference/              # Data sources & formats
│   ├── tutorials/                   # Example notebooks
│   ├── api/                         # API reference
│   ├── development/                 # Contributing guidelines
│   ├── _static/                     # Static assets (CSS, images)
│   └── _templates/                  # Custom Sphinx templates
├── build/                           # Generated HTML (created by make html)
├── Makefile                         # Build commands (Unix/Linux/Mac)
├── make.bat                         # Build commands (Windows)
└── requirements.txt                 # Sphinx dependencies
```

## Documentation Formats

- **RST files** (`.rst`): ReStructuredText format for core structure
- **Markdown files** (`.md`): Markdown format for readability (converted to RST by `myst-parser`)
- **Jupyter notebooks** (`.ipynb`): Executable examples in tutorials

## Common Build Targets

```bash
cd docs

# HTML (most common)
make html

# PDF (requires LaTeX)
make latexpdf

# Clean all build artifacts
make clean

# View all available targets
make help
```

## Editing Documentation

### Markdown Files (.md)

Edit files in `docs/source/*/` directly. Changes are automatically picked up by MyST parser.

Example:
```markdown
# Heading 1

This is a paragraph.

- Bullet point 1
- Bullet point 2

```python
# Code blocks are syntax-highlighted
print("Hello, World!")
```

[Link text](path/to/page.md)
```

### Linking Between Pages

**RST links:**
```rst
:doc:`../user_guide/configuration`
```

**Markdown links:**
```markdown
[Link text](../user_guide/configuration.md)
{doc}`../user_guide/configuration`
```

### Adding Mermaid Diagrams

```markdown
```{mermaid}
flowchart LR
    A --> B
    B --> C
```
```

## Troubleshooting

### Build Fails with "Unknown directive"

Check that all required Sphinx extensions are listed in `conf.py`. Common ones:

```python
extensions = [
    'sphinx.ext.autodoc',      # Auto-document Python code
    'myst_parser',             # Parse Markdown
    'sphinx_design',           # Grid cards
    'sphinxcontrib.mermaid',   # Flowchart diagrams
]
```

### Changes Not Showing Up

Run `make clean` before `make html`:

```bash
cd docs
make clean
make html
```

### Port 8000 Already in Use

Use a different port:

```bash
cd docs/build/html
python -m http.server 8001
```

Then open `http://localhost:8001`

