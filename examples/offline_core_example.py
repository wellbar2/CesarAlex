"""Execute a reproducible CesarAlex workflow without external API requests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cesaralex.lattes import parse_lattes_html  # noqa: E402
from cesaralex.models import Author, DoiResolution, Publication  # noqa: E402
from cesaralex.processor import CesarAlexProcessor  # noqa: E402
from cesaralex.reporting import (  # noqa: E402
    build_application_results,
    dataframe_to_csv_bytes,
)

EXAMPLE_HTML = Path(__file__).with_name("curriculo_lattes_sintetico.html")


class ExampleOrcidProvider:
    """Deterministic public-work response used only by this example."""

    def publications_by_orcid(self, _orcid_url: str) -> list[Publication]:
        return [
            Publication(
                doi="10.5555/cesaralex.1",
                title="Integração de identificadores científicos",
                year=2024,
            )
        ]


class ExampleOpenAlexProvider:
    """Deterministic OpenAlex-like response used only by this example."""

    author = Author(
        openalex_id="A123456789",
        name="Ana Exemplo da Silva",
        orcid="https://orcid.org/0000-0000-0000-0000",
    )

    def publications_by_orcid(self, _orcid_url: str) -> list[Publication]:
        return [
            Publication(
                doi="10.5555/cesaralex.1",
                title="Integração de identificadores científicos",
                year=2024,
                issn_l="1234-5678",
                authors=(self.author,),
            ),
            Publication(
                doi="10.5555/cesaralex.2",
                title="Cobertura de fontes abertas",
                year=2025,
                issn_l="1234-5678",
                authors=(self.author,),
            ),
        ]

    def publications_by_author(self, _author_id: str) -> list[Publication]:
        return self.publications_by_orcid("")

    def resolve_doi(self, _doi: str) -> DoiResolution:
        return DoiResolution.not_found()

    def author_orcid(self, _author_id: str) -> str | None:
        return self.author.orcid


def main(output_path: Path = Path("example_report.csv")) -> Path:
    curriculum = parse_lattes_html(EXAMPLE_HTML.read_text(encoding="utf-8"))
    processor = CesarAlexProcessor(
        ExampleOpenAlexProvider(),
        orcid=ExampleOrcidProvider(),
        sleep_function=lambda _seconds: None,
    )
    outcome = processor.process_curriculum(
        EXAMPLE_HTML.name,
        curriculum,
        check_openalex_by_doi=False,
    )
    results = build_application_results(
        [outcome],
        check_openalex_by_doi=False,
    )
    report = results["all_report_df"]
    output_path.write_bytes(dataframe_to_csv_bytes(report))

    print(f"Researcher: {curriculum.researcher_name}")
    print(f"Rows: {len(report)}")
    print(f"OpenAlex DOIs: {len(outcome.openalex_dois)}")
    print(f"Report: {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    main()
