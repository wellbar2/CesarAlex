# Contributing to CesarAlex

[Versão em português](CONTRIBUTING.md)

Issues and pull requests are welcome. Before proposing a broad change, open an
issue describing the problem, a reproducible example, and the expected
behavior. Changes to matching rules or output-column semantics must be discussed
before implementation.

## Development environment

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Local checks

```bash
python -m ruff format --check cesaralex tests app.py
python -m ruff check cesaralex tests app.py
python -m pytest
python -m pytest --cov=cesaralex
```

Bug fixes should include a regression test. Interface changes should preserve
established labels and output files whenever possible. Do not use real API keys
or personal Lattes data in tests; deterministic providers and synthetic
fixtures are preferred.

Use the public issue tracker to report bugs or ask for help. See
[SUPPORT.md](SUPPORT.md) for the information to include and
[GOVERNANCE.md](GOVERNANCE.md) for the decision process.
