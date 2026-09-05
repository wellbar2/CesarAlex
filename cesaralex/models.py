"""Modelos de domínio independentes da interface Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class Author:
    openalex_id: str
    name: str
    orcid: str | None = None


@dataclass(frozen=True, slots=True)
class Publication:
    doi: str | None
    title: str
    year: str | int
    issn_l: str | None = None
    authors: tuple[Author, ...] = ()


@dataclass(frozen=True, slots=True)
class AuthorMatch:
    openalex_id: str
    name: str
    score: int
    method: str
    orcid: str | None = None


class LookupSource(str, Enum):
    NOT_CHECKED = "nao_consultado"
    NOT_FOUND = "nao_encontrado"
    DOI_API = "doi_openalex"
    DOI_CACHE = "cache_doi"
    INFERRED_PROFILE = "perfil_inferido"


@dataclass(frozen=True, slots=True)
class DoiResolution:
    """Resultado único para qualquer forma de localizar um DOI na OpenAlex."""

    found: bool
    source: LookupSource
    publication: Publication | None = None
    profile_author_id: str | None = None
    profile_author_name: str | None = None
    made_request: bool = False

    @classmethod
    def not_found(
        cls,
        *,
        source: LookupSource = LookupSource.NOT_FOUND,
        made_request: bool = False,
    ) -> DoiResolution:
        return cls(found=False, source=source, made_request=made_request)


@dataclass(frozen=True, slots=True)
class LattesArticle:
    doi: str
    title: str
    year: str


@dataclass(frozen=True, slots=True)
class LattesCurriculum:
    researcher_name: str
    orcid_url: str | None
    articles: dict[str, LattesArticle]

    @property
    def dois(self) -> set[str]:
        return set(self.articles)


@dataclass(slots=True)
class FileProcessingResult:
    filename: str
    researcher_name: str
    orcid_url: str | None
    rows: list[dict[str, Any]]
    lattes_doi_count: int
    openalex_dois: list[str] = field(default_factory=list)
    openalex_detected: bool = False

    @property
    def has_orcid(self) -> bool:
        return bool(self.orcid_url)

    @property
    def has_openalex(self) -> bool:
        return self.openalex_detected or any(
            bool(row.get("Presente na OpenAlex")) for row in self.rows
        )
