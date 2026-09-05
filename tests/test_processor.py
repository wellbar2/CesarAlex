from __future__ import annotations

from cesaralex.models import (
    Author,
    DoiResolution,
    LattesArticle,
    LattesCurriculum,
    LookupSource,
    Publication,
)
from cesaralex.processor import CesarAlexProcessor


class EmptyOrcid:
    def publications_by_orcid(self, _orcid_url: str) -> list[Publication]:
        return []


class DoiOnlyOpenAlex:
    publication = Publication(
        doi="10.2000/example",
        title="Título OpenAlex",
        year=2026,
        issn_l="9876-5432",
        authors=(Author("A9", "Ana Silva", "https://orcid.org/0000-0000-0000-0001"),),
    )

    def publications_by_orcid(self, _orcid_url: str) -> list[Publication]:
        return []

    def publications_by_author(self, _author_id: str) -> list[Publication]:
        return []

    def resolve_doi(self, _doi: str) -> DoiResolution:
        return DoiResolution(
            found=True,
            source=LookupSource.DOI_API,
            publication=self.publication,
            made_request=True,
        )

    def author_orcid(self, _author_id: str) -> str | None:
        return None


def test_doi_fallback_with_orcid_updates_openalex_presence() -> None:
    processor = CesarAlexProcessor(
        DoiOnlyOpenAlex(),
        orcid=EmptyOrcid(),
        sleep_function=lambda _seconds: None,
    )
    curriculum = LattesCurriculum(
        researcher_name="Ana Silva",
        orcid_url="https://orcid.org/0000-0000-0000-0001",
        articles={
            "10.2000/example": LattesArticle("10.2000/example", "Título Lattes", "2026")
        },
    )

    result = processor.process_curriculum(
        "ana.html",
        curriculum,
        check_openalex_by_doi=True,
    )

    assert result.has_openalex is True
    assert result.rows[0]["Presente na OpenAlex"] is True
    assert result.rows[0]["ISSN-L"] == "9876-5432"


def test_curriculum_without_orcid_or_doi_remains_in_report() -> None:
    processor = CesarAlexProcessor(
        DoiOnlyOpenAlex(),
        orcid=EmptyOrcid(),
        sleep_function=lambda _seconds: None,
    )
    curriculum = LattesCurriculum("Sem Produção", None, {})

    result = processor.process_curriculum(
        "vazio.html",
        curriculum,
        check_openalex_by_doi=True,
    )

    assert len(result.rows) == 1
    assert result.rows[0]["DOI"] == "Não disponível"
    assert result.rows[0]["Presente na OpenAlex"] is False
