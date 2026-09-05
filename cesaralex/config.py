"""Configurações internas centralizadas da aplicação."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Parâmetros de acesso às APIs e da heurística de autoria."""

    openalex_doi_timeout: int = 10
    openalex_doi_rate_limit: float = 0.3
    openalex_profile_rate_limit: float = 1.0
    openalex_per_page: int = 200
    openalex_max_pages: int = 25
    openalex_max_retries: int = 5
    openalex_backoff_base: float = 2.0
    enforce_openalex_tokens_subset: bool = True
    min_common_initials_fallback: int = 2


DEFAULT_SETTINGS = Settings()
