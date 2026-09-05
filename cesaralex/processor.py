"""Orquestração da integração Lattes, ORCID e OpenAlex."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from .clients import OpenAlexService, OrcidService
from .config import DEFAULT_SETTINGS, Settings
from .matching import (
    find_author_matches,
    inferred_profile_match,
    matches_to_report,
    should_add_inferred_profile,
)
from .models import (
    AuthorMatch,
    DoiResolution,
    FileProcessingResult,
    LattesCurriculum,
    LookupSource,
    Publication,
)
from .profiles import InferredProfileStore

ProgressCallback = Callable[[int, int, int], None]


def _ignore_progress(_current: int, _total: int, _profile_count: int) -> None:
    return None


class OrcidPublicationsProvider(Protocol):
    def publications_by_orcid(self, orcid_url: str) -> list[Publication]: ...


class OpenAlexProvider(Protocol):
    def publications_by_orcid(self, orcid_url: str) -> list[Publication]: ...

    def publications_by_author(self, author_id: str) -> list[Publication]: ...

    def resolve_doi(self, doi: str) -> DoiResolution: ...

    def author_orcid(self, author_id: str) -> str | None: ...


class CesarAlexProcessor:
    """Serviço testável que não depende da interface Streamlit."""

    def __init__(
        self,
        openalex: OpenAlexProvider,
        *,
        orcid: OrcidPublicationsProvider | None = None,
        settings: Settings = DEFAULT_SETTINGS,
        sleep_function: Callable[[float], None] = time.sleep,
    ) -> None:
        self.openalex = openalex
        self.orcid = orcid or OrcidService()
        self.settings = settings
        self.sleep = sleep_function

    def process_curriculum(
        self,
        filename: str,
        curriculum: LattesCurriculum,
        *,
        check_openalex_by_doi: bool,
        on_doi_progress: ProgressCallback = _ignore_progress,
    ) -> FileProcessingResult:
        if curriculum.orcid_url:
            return self._process_with_orcid(
                filename,
                curriculum,
                check_openalex_by_doi=check_openalex_by_doi,
                on_doi_progress=on_doi_progress,
            )
        return self._process_without_orcid(
            filename,
            curriculum,
            check_openalex_by_doi=check_openalex_by_doi,
            on_doi_progress=on_doi_progress,
        )

    def _process_without_orcid(
        self,
        filename: str,
        curriculum: LattesCurriculum,
        *,
        check_openalex_by_doi: bool,
        on_doi_progress: ProgressCallback,
    ) -> FileProcessingResult:
        if not curriculum.dois:
            return FileProcessingResult(
                filename=filename,
                researcher_name=curriculum.researcher_name,
                orcid_url=None,
                rows=[self._empty_lattes_row(filename, curriculum.researcher_name)],
                lattes_doi_count=0,
            )

        profiles = InferredProfileStore()
        rows: list[dict[str, Any]] = []
        found_dois: set[str] = set()
        dois = sorted(curriculum.dois)

        for current, doi in enumerate(dois, start=1):
            article = curriculum.articles[doi]
            resolution = DoiResolution.not_found(source=LookupSource.NOT_CHECKED)
            matches: list[AuthorMatch] = []

            if check_openalex_by_doi:
                resolution = profiles.resolve(doi) or self.openalex.resolve_doi(doi)
                if resolution.found:
                    found_dois.add(doi)
                    matches = self._matches_for_resolution(
                        curriculum.researcher_name,
                        resolution,
                    )
                    matches = self._complete_match_orcids(matches)
                    self._maybe_add_profile(profiles, resolution, matches)
                if resolution.made_request:
                    self.sleep(self.settings.openalex_doi_rate_limit)
                on_doi_progress(current, len(dois), len(profiles))

            publication = resolution.publication
            title = article.title
            year: str | int = article.year
            if title in {"Não disponível", ""} and publication:
                title = publication.title
            if year in {"Não disponível", ""} and publication:
                year = publication.year

            row = self._base_row(
                filename=filename,
                researcher_name=curriculum.researcher_name,
                orcid="Não disponível",
                doi=doi,
                title=title,
                year=year,
                in_lattes=True,
                in_orcid=False,
                resolution=resolution,
            )
            row.update(matches_to_report(matches))
            row["ORCIDs_Provaveis"] = self._join_match_orcids(matches)
            rows.append(row)

        return FileProcessingResult(
            filename=filename,
            researcher_name=curriculum.researcher_name,
            orcid_url=None,
            rows=rows,
            lattes_doi_count=len(curriculum.dois),
            openalex_dois=sorted(found_dois),
            openalex_detected=bool(found_dois),
        )

    def _process_with_orcid(
        self,
        filename: str,
        curriculum: LattesCurriculum,
        *,
        check_openalex_by_doi: bool,
        on_doi_progress: ProgressCallback,
    ) -> FileProcessingResult:
        assert curriculum.orcid_url is not None
        orcid_publications = self.orcid.publications_by_orcid(curriculum.orcid_url)
        openalex_publications = self.openalex.publications_by_orcid(
            curriculum.orcid_url
        )
        self.sleep(self.settings.openalex_profile_rate_limit)

        orcid_by_doi = _index_by_doi(orcid_publications)
        openalex_by_doi = _index_by_doi(openalex_publications)
        union_dois = sorted(curriculum.dois | set(orcid_by_doi) | set(openalex_by_doi))
        rows: list[dict[str, Any]] = []

        for current, doi in enumerate(union_dois, start=1):
            resolution = self._resolution_from_openalex_profile(
                openalex_by_doi.get(doi)
            )
            if check_openalex_by_doi and not resolution.found:
                resolution = self.openalex.resolve_doi(doi)
                if resolution.made_request:
                    self.sleep(self.settings.openalex_doi_rate_limit)

            preferred = (
                orcid_by_doi.get(doi)
                or curriculum.articles.get(doi)
                or openalex_by_doi.get(doi)
                or resolution.publication
            )
            title = getattr(preferred, "title", "Não disponível")
            year = getattr(preferred, "year", "Não disponível")
            matches: list[AuthorMatch] = []
            if check_openalex_by_doi and resolution.publication:
                matches = find_author_matches(
                    curriculum.researcher_name,
                    resolution.publication.authors,
                    settings=self.settings,
                )
                matches = self._complete_match_orcids(matches)

            row = self._base_row(
                filename=filename,
                researcher_name=curriculum.researcher_name,
                orcid=curriculum.orcid_url,
                doi=doi,
                title=title,
                year=year,
                in_lattes=doi in curriculum.dois,
                in_orcid=doi in orcid_by_doi,
                resolution=resolution,
            )
            row.update(matches_to_report(matches))
            row["ORCIDs_Provaveis"] = self._join_match_orcids(matches)
            rows.append(row)

            if check_openalex_by_doi:
                on_doi_progress(current, len(union_dois), 0)

        found_dois = sorted(
            str(row["DOI"]) for row in rows if bool(row["Presente na OpenAlex"])
        )
        return FileProcessingResult(
            filename=filename,
            researcher_name=curriculum.researcher_name,
            orcid_url=curriculum.orcid_url,
            rows=rows,
            lattes_doi_count=len(curriculum.dois),
            openalex_dois=found_dois,
            openalex_detected=bool(openalex_publications) or bool(found_dois),
        )

    def _matches_for_resolution(
        self,
        researcher_name: str,
        resolution: DoiResolution,
    ) -> list[AuthorMatch]:
        publication = resolution.publication
        if not publication:
            return []
        if (
            resolution.source == LookupSource.INFERRED_PROFILE
            and resolution.profile_author_id
        ):
            return [
                inferred_profile_match(
                    researcher_name,
                    resolution.profile_author_id,
                    resolution.profile_author_name or "",
                    publication.authors,
                )
            ]
        return find_author_matches(
            researcher_name,
            publication.authors,
            settings=self.settings,
        )

    def _maybe_add_profile(
        self,
        profiles: InferredProfileStore,
        resolution: DoiResolution,
        matches: list[AuthorMatch],
    ) -> None:
        if resolution.source == LookupSource.INFERRED_PROFILE:
            return
        if not should_add_inferred_profile(matches):
            return
        match = matches[0]
        if match.openalex_id in profiles:
            return
        publications = self.openalex.publications_by_author(match.openalex_id)
        profiles.add(match.openalex_id, match.name, publications)
        self.sleep(self.settings.openalex_profile_rate_limit)

    def _complete_match_orcids(
        self,
        matches: list[AuthorMatch],
    ) -> list[AuthorMatch]:
        completed: list[AuthorMatch] = []
        for match in matches:
            if match.orcid:
                completed.append(match)
                continue
            orcid = self.openalex.author_orcid(match.openalex_id)
            completed.append(replace(match, orcid=orcid))
            self.sleep(self.settings.openalex_doi_rate_limit)
        return completed

    @staticmethod
    def _join_match_orcids(matches: list[AuthorMatch]) -> str:
        return "; ".join(match.orcid or "-" for match in matches) if matches else "-"

    @staticmethod
    def _resolution_from_openalex_profile(
        publication: Publication | None,
    ) -> DoiResolution:
        if not publication:
            return DoiResolution.not_found(source=LookupSource.NOT_CHECKED)
        return DoiResolution(
            found=True,
            source=LookupSource.DOI_CACHE,
            publication=publication,
            made_request=False,
        )

    @staticmethod
    def _base_row(
        *,
        filename: str,
        researcher_name: str,
        orcid: str,
        doi: str,
        title: str,
        year: str | int,
        in_lattes: bool,
        in_orcid: bool,
        resolution: DoiResolution,
    ) -> dict[str, Any]:
        publication = resolution.publication
        return {
            "Arquivo": filename,
            "nomeLattes": researcher_name,
            "ORCID": orcid,
            "DOI": doi,
            "Título": title,
            "Ano": year,
            "Presente no Lattes": in_lattes,
            "Presente no ORCID": in_orcid,
            "Presente na OpenAlex": resolution.found,
            "ISSN-L": (
                publication.issn_l
                if publication and publication.issn_l
                else "Sem ISSN-L"
                if resolution.found
                else "Não disponível"
            ),
        }

    @staticmethod
    def _empty_lattes_row(filename: str, researcher_name: str) -> dict[str, Any]:
        return {
            "Arquivo": filename,
            "nomeLattes": researcher_name,
            "ORCID": "Não disponível",
            "DOI": "Não disponível",
            "Título": "Não disponível",
            "Ano": "Não disponível",
            "Presente no Lattes": True,
            "Presente no ORCID": False,
            "Presente na OpenAlex": False,
            "ISSN-L": "Não disponível",
            "MatchPairs_OpenAlex": "-",
            "IdsOpenAlex_Provaveis": "-",
            "NomesOpenAlex_Provaveis": "-",
            "ORCIDs_Provaveis": "-",
            "MetodoMatch": "-",
            "MatchScoreMax": 0,
        }


def _index_by_doi(publications: list[Publication]) -> dict[str, Publication]:
    return {
        publication.doi: publication for publication in publications if publication.doi
    }


__all__ = [
    "CesarAlexProcessor",
    "OpenAlexService",
    "OrcidService",
]
