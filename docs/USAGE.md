# Usage guide

## Graphical workflow

1. Save each public Lattes curriculum page as a complete HTML file.
2. Start CesarAlex with `streamlit run app.py`.
3. Enter an OpenAlex API key in the sidebar. Separate multiple keys with `;`.
4. Decide whether to enable DOI lookup for curricula without ORCID.
5. Upload one or more Lattes HTML files.
6. Select **Processar currículos**.
7. Review the coverage visualizations and download the individual and
   consolidated CSV files.

The interface is in Brazilian Portuguese because its terminology follows the
Lattes Platform. The central controls translate as follows:

| Interface label | Meaning |
| --- | --- |
| `Chave OpenAlex` | OpenAlex API key |
| `Verificar OpenAlex mesmo sem ORCID (por DOI)` | Search OpenAlex by DOI when an ORCID is unavailable |
| `Atualizar saldos` | Refresh OpenAlex balances |
| `Processar currículos` | Process curricula |
| `Baixar relatório completo em CSV` | Download the consolidated CSV report |

## Choosing the DOI option

When the option is disabled, curricula with ORCID are integrated through their
ORCID profiles, and curricula without ORCID remain represented without an
OpenAlex DOI search.

When it is enabled, CesarAlex may query individual DOIs that were not recovered
from an OpenAlex profile. For a curriculum without ORCID, a strong and unique
name match can identify an OpenAlex author. CesarAlex then downloads that
author's profile once and searches subsequent DOIs in the local profile cache.
If the next DOI is absent from all inferred profiles, the application queries
OpenAlex again and may add another strongly matched profile.

This mode improves recovery but may increase processing time and API usage.
Monitor the sidebar before running a large collection.

## Reproducible network-free example

The repository contains a synthetic Lattes-like HTML file and deterministic
provider responses. They exercise parsing, DOI normalization, source
consolidation, and CSV generation without contacting ORCID or OpenAlex:

```bash
python examples/offline_core_example.py
```

The command writes `example_report.csv` in the current directory and prints a
short coverage summary. All identifiers and people in the example are
synthetic.

## Reading the outputs

The unit of the consolidated report is a researcher-publication pair. Important
fields are:

| Column | Interpretation |
| --- | --- |
| `Arquivo` | Uploaded HTML filename |
| `nomeLattes` | Researcher name extracted from Lattes |
| `ORCID` | ORCID found in the curriculum, if available |
| `DOI` | Normalized lowercase DOI |
| `Presente no Lattes` | DOI was extracted from the Lattes HTML |
| `Presente no ORCID` | DOI was returned by the ORCID record |
| `Presente na OpenAlex` | DOI was recovered from an OpenAlex profile, lookup, or inferred-profile cache |
| `ISSN-L` | Linking ISSN returned by OpenAlex, when available |
| `IdsOpenAlex_Provaveis` | Candidate OpenAlex author IDs from name matching |
| `NomesOpenAlex_Provaveis` | Display names for the candidate IDs |
| `ORCIDs_Provaveis` | ORCIDs associated with matched OpenAlex authors |
| `MetodoMatch` | Matching route, such as exact token match or inferred profile |
| `MatchScoreMax` | Maximum conservative name-overlap score |

The application exports semicolon-delimited UTF-8 CSV and quotes every field.
Boolean columns indicate source presence and should not be interpreted as a
quality judgment about the researcher or source.

## OpenAlex balances

The sidebar reports processed curricula, daily balance, and prepaid balance.
The cost shown during a run is obtained from response metadata when supplied by
OpenAlex. Balances are refreshed before and after processing; intermediate bar
changes are estimates until the final refresh.

## Reproducible research record

Record the following in any study that uses CesarAlex:

- CesarAlex version or Git commit;
- collection date;
- selected DOI option;
- input population and inclusion rules;
- OpenAlex and ORCID access dates;
- any manual corrections or post-processing;
- the exported consolidated CSV and an integrity checksum.

Never store API keys in notebooks, CSV files, issue reports, or repositories.
