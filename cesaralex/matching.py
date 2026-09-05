"""Heurística de associação entre o nome Lattes e autores OpenAlex."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .config import DEFAULT_SETTINGS, Settings
from .models import Author, AuthorMatch
from .normalization import normalize_text

RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)
RE_INITIAL = re.compile(
    r"\b([A-Za-zÀ-ÖØ-öø-ÿ])\.\b|\b([A-Za-zÀ-ÖØ-öø-ÿ])\b",
    flags=re.UNICODE,
)
STOP_TOKENS = {
    "de",
    "da",
    "do",
    "das",
    "dos",
    "del",
    "la",
    "las",
    "los",
    "el",
    "y",
    "e",
    "van",
    "von",
}


def name_tokens(name: str) -> list[str]:
    tokens = RE_WORD.findall(normalize_text(name))
    return [token for token in tokens if len(token) >= 2 and token not in STOP_TOKENS]


def name_initials(name: str) -> set[str]:
    normalized = normalize_text(name)
    initials = {token[0] for token in name_tokens(normalized) if token}
    for match in RE_INITIAL.finditer(normalized):
        initial = match.group(1) or match.group(2)
        if initial:
            initials.add(normalize_text(initial)[0])
    return initials


def overlap_score(lattes_name: str, openalex_name: str) -> int:
    return len(set(name_tokens(lattes_name)).intersection(name_tokens(openalex_name)))


def openalex_tokens_are_subset(lattes_name: str, openalex_name: str) -> bool:
    lattes_tokens = set(name_tokens(lattes_name))
    openalex_tokens = set(name_tokens(openalex_name))
    return not openalex_tokens or openalex_tokens.issubset(lattes_tokens)


def find_author_matches(
    lattes_name: str,
    authors: Iterable[Author],
    *,
    min_common_tokens: int = 2,
    settings: Settings = DEFAULT_SETTINGS,
) -> list[AuthorMatch]:
    """Aplica a regra principal e, se necessário, o fallback por iniciais."""

    authors = tuple(authors)
    primary = _best_matches_by_id(
        lattes_name,
        authors,
        method="tokens>=2",
        score_function=lambda author: overlap_score(lattes_name, author.name),
        accept_function=lambda score, _author: score >= min_common_tokens,
    )
    primary = _apply_subset_filter(lattes_name, primary, settings)
    if primary:
        return primary

    lattes_initials = name_initials(lattes_name)

    def fallback_accept(score: int, author: Author) -> bool:
        common_initials = lattes_initials.intersection(name_initials(author.name))
        return (
            score >= 1 and len(common_initials) >= settings.min_common_initials_fallback
        )

    fallback = _best_matches_by_id(
        lattes_name,
        authors,
        method="fallback_1token+2iniciais",
        score_function=lambda author: overlap_score(lattes_name, author.name),
        accept_function=fallback_accept,
    )
    return _apply_subset_filter(lattes_name, fallback, settings)


def inferred_profile_match(
    lattes_name: str,
    author_id: str,
    author_name: str,
    authors: Iterable[Author],
) -> AuthorMatch:
    """Cria o match de um perfil já validado usando o nome real do autor."""

    owner = next(
        (author for author in authors if author.openalex_id == author_id), None
    )
    resolved_name = owner.name if owner and owner.name else author_name
    resolved_orcid = owner.orcid if owner else None
    return AuthorMatch(
        openalex_id=author_id,
        name=resolved_name or author_id,
        score=overlap_score(lattes_name, resolved_name or ""),
        method="perfil_inferido",
        orcid=resolved_orcid,
    )


def should_add_inferred_profile(matches: list[AuthorMatch]) -> bool:
    """Aceita automaticamente apenas um match principal único e forte."""

    return (
        len(matches) == 1 and matches[0].method == "tokens>=2" and matches[0].score >= 2
    )


def matches_to_report(matches: list[AuthorMatch]) -> dict[str, str | int]:
    if not matches:
        return {
            "MatchPairs_OpenAlex": "-",
            "IdsOpenAlex_Provaveis": "-",
            "NomesOpenAlex_Provaveis": "-",
            "MetodoMatch": "-",
            "MatchScoreMax": 0,
        }
    return {
        "MatchPairs_OpenAlex": "; ".join(
            f"{match.openalex_id}|{match.name}" for match in matches
        ),
        "IdsOpenAlex_Provaveis": "; ".join(match.openalex_id for match in matches),
        "NomesOpenAlex_Provaveis": "; ".join(match.name for match in matches),
        "MetodoMatch": "; ".join(match.method for match in matches),
        "MatchScoreMax": max(match.score for match in matches),
    }


def _best_matches_by_id(
    lattes_name: str,
    authors: tuple[Author, ...],
    *,
    method: str,
    score_function,
    accept_function,
) -> list[AuthorMatch]:
    del lattes_name  # explicit: the closures above own the comparison rule
    best_by_id: dict[str, AuthorMatch] = {}
    for author in authors:
        if not author.openalex_id or not author.name:
            continue
        score = score_function(author)
        if not accept_function(score, author):
            continue
        candidate = AuthorMatch(
            openalex_id=author.openalex_id,
            name=author.name,
            score=score,
            method=method,
            orcid=author.orcid,
        )
        previous = best_by_id.get(author.openalex_id)
        if (
            previous is None
            or score > previous.score
            or (score == previous.score and len(author.name) < len(previous.name))
        ):
            best_by_id[author.openalex_id] = candidate
    return sorted(
        best_by_id.values(),
        key=lambda match: (
            -match.score,
            len(match.name),
            normalize_text(match.name),
            match.openalex_id,
        ),
    )


def _apply_subset_filter(
    lattes_name: str,
    matches: list[AuthorMatch],
    settings: Settings,
) -> list[AuthorMatch]:
    if not settings.enforce_openalex_tokens_subset:
        return matches
    return [
        match
        for match in matches
        if openalex_tokens_are_subset(lattes_name, match.name)
    ]
