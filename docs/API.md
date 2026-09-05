# Core Python API

The supported programmatic surface is the set of classes and functions below.
Objects whose names begin with `_` are internal implementation details.

## Parsing and normalization

### `cesaralex.lattes.decode_lattes_html(data)`

Decodes exported HTML bytes, trying UTF-8 variants and Latin-1. Returns `str`.

### `cesaralex.lattes.parse_lattes_html(content)`

Parses a complete Lattes HTML string once and returns `LattesCurriculum` with
the researcher name, optional ORCID URL, and DOI-indexed articles.

### `cesaralex.normalization.normalize_doi(value)`

Removes common DOI URL and textual prefixes, normalizes case, and returns either
the canonical DOI string or `None` for invalid/missing input.

### Other identifier helpers

- `extract_orcid_id(orcid_url)` returns the bare ORCID;
- `normalize_openalex_author_id(author_id)` returns the bare OpenAlex author ID;
- `normalize_text(value)` prepares Unicode names for matching.

## Domain models

`cesaralex.models` defines immutable `Author`, `Publication`, `AuthorMatch`,
`DoiResolution`, `LattesArticle`, and `LattesCurriculum` records. A
`FileProcessingResult` contains all report rows and curriculum-level flags.
`LookupSource` records whether a DOI came from direct lookup, request cache,
inferred profile, or was not found/checked.

## External services

### `OpenAlexClient(api_keys, ...)`

Low-level JSON client with bounded retries, ordered key rotation, request
classification, and reported-cost accumulation. `get_json()` accepts an
OpenAlex endpoint or URL. `get_rate_limit()` reads account balances, and
`usage_stats()` returns counters for the current client.

### `OpenAlexService(client, ...)`

Domain adapter with these main methods:

- `publications_by_orcid(orcid_url)`;
- `publications_by_author(author_id)`;
- `resolve_doi(doi)`;
- `author_orcid(author_id)`.

DOI and author resolutions are cached for the lifetime of the service.

### `OrcidService(...)`

Retrieves public ORCID work summaries with `publications_by_orcid(orcid_url)`.

## Processing

### `CesarAlexProcessor(openalex, orcid=..., settings=...)`

Coordinates the complete integration. Its primary method is:

```python
result = processor.process_curriculum(
    filename="curriculo.html",
    curriculum=curriculum,
    check_openalex_by_doi=True,
    on_doi_progress=callback,
)
```

The OpenAlex object may be `OpenAlexService` or any object that satisfies the
`OpenAlexProvider` protocol. This enables deterministic tests and alternative
adapters. The ORCID object follows `OrcidPublicationsProvider`.

## Matching

`find_author_matches(lattes_name, authors, settings=...)` returns scored
`AuthorMatch` objects. `should_add_inferred_profile(matches)` applies the
conservative acceptance rule, and `matches_to_report(matches)` creates the
auditable CSV fields.

## Reports

`build_application_results(outcomes, check_openalex_by_doi=...)` creates
per-file and consolidated data frames, warning tables, and visualization
counts. `dataframe_to_csv_bytes(dataframe)` applies the project's stable CSV
format. `compute_venn_counts(rows)` returns exclusive three-source coverage
counts.

## Usage balances

`OpenAlexBalance` represents aggregated daily and prepaid balances.
`balance_from_rate_limit_payload(payload)` parses one key response, while
`aggregate_balances(balances)` combines several keys. `estimate_after_cost()`
provides an intermediate display estimate between authoritative balance reads.

## Example

See [examples/offline_core_example.py](../examples/offline_core_example.py) for
a complete programmatic workflow with synthetic data and no network access.
