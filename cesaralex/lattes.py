"""Leitura de currículos Lattes exportados em HTML."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup, Tag

from .models import LattesArticle, LattesCurriculum
from .normalization import normalize_doi


def parse_lattes_html(content: str) -> LattesCurriculum:
    """Extrai nome, ORCID e artigos com DOI fazendo um único parse do HTML."""

    soup = BeautifulSoup(content, "html.parser")
    return LattesCurriculum(
        researcher_name=_extract_researcher_name(soup),
        orcid_url=_extract_orcid_url(soup),
        articles=_extract_articles(soup),
    )


def decode_lattes_html(data: bytes) -> str:
    """Decodifica o HTML tolerando exportações antigas do Lattes."""

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_researcher_name(soup: BeautifulSoup) -> str:
    heading = soup.find("h2", class_="nome")
    if heading:
        name = heading.get_text(" ", strip=True)
        if name:
            return name
    return "Não disponível"


def _extract_orcid_url(soup: BeautifulSoup) -> str | None:
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))
        if "orcid.org" in href:
            return href.strip()
    return None


def _extract_articles(soup: BeautifulSoup) -> dict[str, LattesArticle]:
    articles: dict[str, LattesArticle] = {}
    for link in soup.find_all("a", class_="icone-producao icone-doi"):
        doi = normalize_doi(str(link.get("href", "")))
        if not doi:
            continue
        parent = link.find_parent("span", class_="transform") or link.find_parent()
        parent_text = parent.get_text(" ", strip=True) if parent else ""
        year_match = re.search(r"\b(?:19|20)\d{2}\b", parent_text)
        articles[doi] = LattesArticle(
            doi=doi,
            title=_extract_article_title(link),
            year=year_match.group(0) if year_match else "Não disponível",
        )
    return articles


def _extract_article_title(link: Tag) -> str:
    block = link.find_parent("span", class_="transform") or link.find_parent()
    if not block:
        return "Não disponível"

    cited = block.find("span", class_="citado")
    if cited:
        cvuri = str(cited.get("cvuri", ""))
        if cvuri and "titulo=" in cvuri:
            title = (parse_qs(urlparse(cvuri).query).get("titulo") or [""])[0].strip()
            if title:
                return title

    clean_block = BeautifulSoup(str(block), "html.parser")
    for selector in (
        "span.informacao-artigo",
        "a.icone-producao.icone-doi",
        "span.citado",
        "sup",
    ):
        for tag in clean_block.select(selector):
            tag.decompose()

    text = re.sub(r"\s+", " ", clean_block.get_text(" ", strip=True)).strip()
    after_authors = text.split(" . ", 1)[1].strip() if " . " in text else text
    match = re.match(
        r"^(.+?)\.\s+[^.]*,\s*(v\.|p\.|n\.|ano|20\d{2}|19\d{2})",
        after_authors,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return after_authors or text or "Não disponível"
