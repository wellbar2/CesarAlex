"""Cache de perfis OpenAlex inferidos para reduzir consultas DOI a DOI."""

from __future__ import annotations

from dataclasses import dataclass

from .models import DoiResolution, LookupSource, Publication
from .normalization import normalize_doi, normalize_openalex_author_id


@dataclass(frozen=True, slots=True)
class InferredProfile:
    author_id: str
    author_name: str
    publications_by_doi: dict[str, Publication]


class InferredProfileStore:
    """Índice local dos perfis validados durante um currículo."""

    def __init__(self) -> None:
        self._profiles: dict[str, InferredProfile] = {}

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, author_id: object) -> bool:
        return normalize_openalex_author_id(str(author_id)) in self._profiles

    def add(
        self,
        author_id: str,
        author_name: str,
        publications: list[Publication],
    ) -> None:
        author_id = normalize_openalex_author_id(author_id)
        if not author_id:
            return
        publications_by_doi = {
            publication.doi: publication
            for publication in publications
            if publication.doi
        }
        self._profiles[author_id] = InferredProfile(
            author_id=author_id,
            author_name=author_name,
            publications_by_doi=publications_by_doi,
        )

    def resolve(self, doi: str) -> DoiResolution | None:
        """Localiza o DOI sem rede e conserva a identidade do perfil proprietário."""

        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            return None
        for profile in self._profiles.values():
            publication = profile.publications_by_doi.get(normalized_doi)
            if publication:
                return DoiResolution(
                    found=True,
                    source=LookupSource.INFERRED_PROFILE,
                    publication=publication,
                    profile_author_id=profile.author_id,
                    profile_author_name=profile.author_name,
                    made_request=False,
                )
        return None
