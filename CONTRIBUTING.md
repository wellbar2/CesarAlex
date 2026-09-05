# Contribuindo com o CesarAlex

[English version](CONTRIBUTING.en.md)

Contribuições por *issues* e *pull requests* são bem-vindas. Antes de propor uma
mudança ampla, abra uma *issue* descrevendo o problema, um exemplo reproduzível e
o comportamento esperado.

## Ambiente de desenvolvimento

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Verificações locais

```bash
python -m ruff format --check cesaralex tests examples app.py
python -m ruff check cesaralex tests examples app.py
python -m pytest
python -m pytest --cov=cesaralex
```

Novas correções devem incluir um teste de regressão. Mudanças na interface devem
preservar, sempre que possível, os rótulos e os arquivos de saída já utilizados
pela comunidade.

Para pedir ajuda ou relatar um defeito, utilize o rastreador de *issues* do
repositório público. Consulte [SUPPORT.md](SUPPORT.md) para saber quais
informações incluir e como evitar a exposição de chaves ou dados pessoais.

Alterações que modifiquem regras de correspondência ou o significado das
colunas de saída devem ser discutidas antes em uma *issue*. O processo de
decisão está descrito em [GOVERNANCE.md](GOVERNANCE.md).
