# CesarAlex

[Documentação em português](README.md) · Version declared in the source: **1.0.0**

CesarAlex is a Streamlit application and a reusable Python core for integrating
and comparing publication records from Brazilian **Lattes curricula**, **ORCID**
profiles, and **OpenAlex**. DOI is used as the primary comparison key. The
outputs show, for every researcher-publication pair, in which sources the DOI
was found.

Public demonstration: <https://cesaralex-beta.streamlit.app/>

## Statement of need

The Lattes Platform is the central curriculum information system for the
Brazilian research community, whereas ORCID and OpenAlex provide internationally
interoperable identifiers and metadata. Querying these systems independently
makes it difficult to determine which publications are present, absent, or
exclusive to each source for a given researcher. CesarAlex automates this
integration and produces auditable data for bibliometric coverage studies,
identifier curation, and assessment of academic-profile completeness.

The intended users are bibliometrics and scientometrics researchers,
librarians, research managers, and teams working with Lattes data. The
comparison with related software and the research context are described in the
[JOSS paper](paper/paper.md).

## Main features

- simultaneous ingestion of one or more Lattes curricula exported as HTML;
- extraction of researcher name, ORCID, DOI, title, and year from Lattes;
- publication retrieval through the public ORCID API;
- cursor-paginated publication retrieval from OpenAlex by ORCID;
- optional DOI lookup for curricula without an ORCID;
- conservative OpenAlex author inference through name matching;
- inferred-profile caching to avoid repeated DOI requests;
- per-researcher and consolidated CSV reports;
- identification of curricula without ORCID whose publications were found in
  OpenAlex;
- Sankey and three-source coverage visualizations;
- OpenAlex daily and prepaid balance monitoring, including multiple API keys.

## Installation

CesarAlex requires Python 3.10 or newer.

```bash
git clone https://github.com/wellbar2/CesarAlex.git
cd CesarAlex
python -m venv .venv
```

Activate the environment before installing: `.venv\Scripts\activate` in
Windows Command Prompt, `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS.

```bash
python -m pip install -e .
```


## Running the application

```bash
streamlit run app.py
```

In the sidebar, enter one or more OpenAlex API keys separated by semicolons.
Upload the Lattes HTML files and select **Processar currículos**. The interface
is intentionally presented in Brazilian Portuguese because it follows the
terminology used by the Lattes Platform.

The option **Verificar OpenAlex mesmo sem ORCID (por DOI)** enables individual
DOI lookup and inferred-profile caching. This can increase processing time and
API usage.

A detailed walkthrough and a network-free example are available in
[docs/USAGE.md](docs/USAGE.md). The synthetic file
[examples/curriculo_lattes_sintetico.html](examples/curriculo_lattes_sintetico.html)
contains no personal data.

## OpenAlex usage indicators

The sidebar contains three progress bars:

1. processed curricula;
2. remaining daily balance, daily budget, and daily spending;
3. remaining prepaid balance.

The interface also reports HTTP calls, list and singleton queries, and the cost
reported by OpenAlex during the run. Balances can be refreshed manually and are
queried automatically before and after processing. If several valid keys are
provided, their balances are aggregated without exposing the keys in reports.

## Architecture

The Streamlit layer is separated from the testable domain core:

```text
app.py                  Streamlit entry point
cesaralex/
  clients.py            ORCID/OpenAlex clients and request-level caches
  config.py             centralized internal settings
  lattes.py             Lattes HTML parser
  matching.py           author-matching heuristics
  models.py             domain models
  processor.py          interface-independent processing workflow
  profiles.py           inferred virtual-profile cache
  reporting.py          CSV and coverage result preparation
  ui.py                 Streamlit interface
  usage.py              balance, cost, and multiple-key aggregation
  visualizations.py     Sankey and coverage diagram
tests/                  automated tests
docs/                   user and technical documentation
examples/               synthetic and network-free examples
paper/                  JOSS paper and bibliography
```

Design decisions and trade-offs are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and the reusable Python API is
summarized in [docs/API.md](docs/API.md).

## Tests and quality assurance

```bash
python -m pip install -e ".[dev]"
python -m ruff format --check cesaralex tests examples app.py
python -m ruff check cesaralex tests examples app.py
python -m pytest
python -m pytest --cov=cesaralex
```

The tests use deterministic providers and simulated API responses, so no
OpenAlex key or network access is needed. They cover HTML parsing, identifier
normalization, author matching, report generation, balance calculations,
request accounting, Streamlit startup, and the inferred-profile regression.
See [docs/VALIDATION.md](docs/VALIDATION.md).

## Main outputs

CSV files use semicolon delimiters, UTF-8 encoding, and double quotes for all
fields. Important columns include:

- `Arquivo`, `nomeLattes`, `ORCID`, `DOI`, `Título`, `Ano`;
- `Presente no Lattes`, `Presente no ORCID`, `Presente na OpenAlex`;
- `ISSN-L`;
- `MatchPairs_OpenAlex`, `IdsOpenAlex_Provaveis`,
  `NomesOpenAlex_Provaveis`, `ORCIDs_Provaveis`, `MetodoMatch`, and
  `MatchScoreMax`.

## Limitations and responsible use

- publications without a DOI are outside the main comparison unit;
- integration quality depends on identifiers present in the source systems;
- author inference is deliberately conservative but remains heuristic;
- external API availability and usage limits can affect processing;
- the hosted demonstration has fewer resources than a local installation.

Lattes files may contain personal data. Read
[docs/PRIVACY.md](docs/PRIVACY.md) before processing or sharing third-party
curricula.

## Documentation and community

- [Detailed installation](docs/INSTALLATION.md)
- [Usage guide](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Core API](docs/API.md)
- [Validation](docs/VALIDATION.md)
- [JOSS readiness](docs/JOSS_READINESS.md)
- [Contributing](CONTRIBUTING.md)
- [Governance](GOVERNANCE.md)
- [Support](SUPPORT.md)

## Authors and citation

- Wellington Barbosa Rodrigues — software development and technical
  responsibility;
- Thamyres Vieira dos Santos — conception and domain guidance;
- Rogério Mugnaini — conception, domain guidance, and project direction.

Citation metadata are provided in [CITATION.cff](CITATION.cff), and contributor
roles are described in [AUTHORS.md](AUTHORS.md).

## License

CesarAlex is distributed under the GNU General Public License v3.0 only. See
[LICENSE](LICENSE).

## Scientific publications / Publicações científicas

- **Origin of the research approach:** Rodrigues, Santos, and Mugnaini (2026), *BiblioS*, [DOI: 10.5195/biblios.2026.1370](https://doi.org/10.5195/biblios.2026.1370). The study developed the scientific basis for the three-source DOI coverage analysis; do not assume it was run with the current release.
- **Confirmed application of CesarAlex:** Rodrigues, Tomassini, Mugnaini, and Mena-Chalco (2026), *Além do ORCID autodeclarado no Lattes: inferência de identificadores via DOI–OpenAlex e seus limites*, X EBBC, [DOI: 10.22477/x.ebbc.1017](https://doi.org/10.22477/x.ebbc.1017). The project lead confirms that this study used CesarAlex. The reported inference rates are specific to the studied population and should not be presented as general software accuracy.


## Validation evidence

On 29 October 2026 the supplied snapshot was installed in a clean Linux
environment with Python 3.12.14. All 28 tests passed, including Streamlit startup.
Ruff checks, dependency checks, and distribution builds also passed.
These checks use controlled responses; original-curriculum and authenticated
network validation remain pending. See the
[execution record](docs/RESULTADOS_VALIDACAO_2026-10-29.md) and
[real-data validation guide, in Portuguese](docs/VALIDACAO_REAL_PASSO_A_PASSO.md).
