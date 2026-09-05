"""Normalização de identificadores e texto."""

from __future__ import annotations

import re
import unicodedata

DOI_PREFIXES = (
    "http://dx.doi.org/",
    "https://dx.doi.org/",
    "https://doi.org/",
    "http://doi.org/",
    "doi:",
)


def normalize_doi(doi: str | None) -> str | None:
    """Remove prefixos usuais do DOI e devolve uma chave em minúsculas."""

    if not doi:
        return None
    normalized = doi.strip().lower()
    if normalized in {"sem doi", "-", "na", "n/a"}:
        return None
    for prefix in DOI_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).strip()
            break
    return normalized or None


def normalize_openalex_author_id(author_id: str | None) -> str:
    """Converte URL ou ID de autor OpenAlex para a forma ``A...``."""

    if not author_id:
        return ""
    return author_id.strip().removeprefix("https://openalex.org/").strip()


def extract_orcid_id(orcid_url: str | None) -> str:
    """Extrai o identificador ORCID de uma URL."""

    if not orcid_url:
        return ""
    return (
        orcid_url.replace("https://orcid.org/", "")
        .replace("http://orcid.org/", "")
        .strip()
    )


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(value: str) -> str:
    value = strip_accents(value or "").casefold()
    value = re.sub(r"[-‐-―]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()
