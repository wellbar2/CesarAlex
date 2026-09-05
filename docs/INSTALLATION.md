# Installation

## Requirements

- Python 3.10, 3.11, or 3.12;
- internet access for real ORCID and OpenAlex queries;
- an OpenAlex API key for application processing;
- a current browser for the Streamlit interface.

No ORCID credential is required for the public records queried by CesarAlex.

## Install from a repository clone

```bash
git clone https://github.com/wellbar2/CesarAlex.git
cd CesarAlex
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the package and runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Start the interface:

```bash
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## Development installation

```bash
python -m pip install -e ".[dev]"
python -m ruff format --check cesaralex tests examples app.py
python -m ruff check cesaralex tests examples app.py
python -m pytest
```

The automated tests do not require API keys or network access.

## Build the distribution

```bash
python -m build
```

This creates source and wheel distributions under `dist/`. The version in
`pyproject.toml`, `cesaralex/__init__.py`, and `CITATION.cff` must agree before a
release.

## Troubleshooting

### `streamlit` is not recognized

Use `python -m streamlit run app.py` or confirm that the virtual environment is
active.

### The processing button is disabled

At least one HTML file and one OpenAlex API key are required. Multiple keys must
be separated by semicolons.

### External requests fail

Confirm internet access and the OpenAlex key. Temporary ORCID or OpenAlex
failures are reported in the interface. Retry only after checking the message so
that unnecessary paid requests are not generated.

### A Lattes file is not parsed as expected

Export or save the complete public curriculum page as HTML. Open an issue with a
minimal synthetic example; do not publish a third party's complete curriculum.
