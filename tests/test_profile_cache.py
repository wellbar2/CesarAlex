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


class ProfileAwareOpenAlex:
    def __init__(self) -> None:
        self.doi_calls: list[str] = []
        self.profile_calls: list[str] = []
        self.owner = Author(
            openalex_id="A123",
            name="Wellington Barbosa Rodrigues",
            orcid="https://orcid.org/0000-0002-6465-6012",
        )
        self.first = Publication(
            doi="10.1000/a",
            title="Primeiro trabalho",
            year=2024,
            issn_l="1234-5678",
            authors=(self.owner,),
        )
        self.second = Publication(
            doi="10.1000/b",
            title="Segundo trabalho",
            year=2025,
            issn_l="1234-5678",
            authors=(self.owner,),
        )

    def publications_by_orcid(self, _orcid_url: str) -> list[Publication]:
        return []

    def publications_by_author(self, author_id: str) -> list[Publication]:
        self.profile_calls.append(author_id)
        return [self.first, self.second]

    def resolve_doi(self, doi: str) -> DoiResolution:
        self.doi_calls.append(doi)
        if doi != self.first.doi:
            raise AssertionError(
                "O segundo DOI deveria ser resolvido no perfil virtual"
            )
        return DoiResolution(
            found=True,
            source=LookupSource.DOI_API,
            publication=self.first,
            made_request=True,
        )

    def author_orcid(self, _author_id: str) -> str | None:
        raise AssertionError("O ORCID já está no authorship e não exige consulta")


def test_profile_hit_marks_openalex_and_preserves_real_author_identity() -> None:
    openalex = ProfileAwareOpenAlex()
    processor = CesarAlexProcessor(
        openalex,
        orcid=EmptyOrcid(),
        sleep_function=lambda _seconds: None,
    )
    curriculum = LattesCurriculum(
        researcher_name="Wellington Barbosa Rodrigues",
        orcid_url=None,
        articles={
            "10.1000/a": LattesArticle("10.1000/a", "Primeiro trabalho", "2024"),
            "10.1000/b": LattesArticle("10.1000/b", "Segundo trabalho", "2025"),
        },
    )

    result = processor.process_curriculum(
        "curriculo.html",
        curriculum,
        check_openalex_by_doi=True,
    )

    cached_row = next(row for row in result.rows if row["DOI"] == "10.1000/b")
    assert cached_row["Presente na OpenAlex"] is True
    assert cached_row["IdsOpenAlex_Provaveis"] == "A123"
    assert cached_row["NomesOpenAlex_Provaveis"] == "Wellington Barbosa Rodrigues"
    assert cached_row["MatchPairs_OpenAlex"] == ("A123|Wellington Barbosa Rodrigues")
    assert cached_row["MetodoMatch"] == "perfil_inferido"
    assert cached_row["MatchScoreMax"] == 3
    assert "inferido anteriormente" not in str(cached_row)
    assert openalex.doi_calls == ["10.1000/a"]
    assert openalex.profile_calls == ["A123"]


def test_progress_reports_profile_count_without_extra_doi_request() -> None:
    openalex = ProfileAwareOpenAlex()
    processor = CesarAlexProcessor(
        openalex,
        orcid=EmptyOrcid(),
        sleep_function=lambda _seconds: None,
    )
    curriculum = LattesCurriculum(
        researcher_name="Wellington Barbosa Rodrigues",
        orcid_url=None,
        articles={
            doi: LattesArticle(doi, doi, "2025") for doi in ("10.1000/a", "10.1000/b")
        },
    )
    progress: list[tuple[int, int, int]] = []

    processor.process_curriculum(
        "curriculo.html",
        curriculum,
        check_openalex_by_doi=True,
        on_doi_progress=lambda current, total, profiles: progress.append(
            (current, total, profiles)
        ),
    )

    assert progress == [(1, 2, 1), (2, 2, 1)]
