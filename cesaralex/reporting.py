"""Montagem dos relatórios, contadores e arquivos CSV."""

from __future__ import annotations

import csv
import hashlib
from typing import Any

import pandas as pd

from .models import FileProcessingResult

TEXT_COLUMNS = [
    "Arquivo",
    "ORCID",
    "DOI",
    "Título",
    "ISSN-L",
    "Motivo",
    "Exemplos_DOI_OpenAlex",
    "nomeLattes",
    "MatchPairs_OpenAlex",
    "IdsOpenAlex_Provaveis",
    "NomesOpenAlex_Provaveis",
    "ORCIDs_Provaveis",
    "MetodoMatch",
]
DISPLAY_STRING_COLUMNS = [
    "Ano",
    "Presente no Lattes",
    "Presente no ORCID",
    "Presente na OpenAlex",
    "MatchScoreMax",
]


def build_application_results(
    outcomes: list[FileProcessingResult],
    *,
    check_openalex_by_doi: bool,
) -> dict[str, Any]:
    """Consolida os resultados no formato mantido pela sessão Streamlit."""

    all_rows = [row for outcome in outcomes for row in outcome.rows]
    no_orcid_with_openalex = [
        outcome
        for outcome in outcomes
        if not outcome.has_orcid and outcome.openalex_dois
    ]
    return {
        "per_file_df": {
            outcome.filename: pd.DataFrame(outcome.rows) for outcome in outcomes
        },
        "all_report_df": pd.DataFrame(all_rows),
        "warn_files_df": pd.DataFrame(
            [_warning_file_row(outcome) for outcome in no_orcid_with_openalex]
        ),
        "warn_rows_df": pd.DataFrame(
            row
            for outcome in no_orcid_with_openalex
            for row in _warning_doi_rows(outcome)
        ),
        "sankey": _sankey_counts(outcomes),
        "options": {"check_openalex_without_orcid": check_openalex_by_doi},
    }


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Gera CSV UTF-8, separado por ponto e vírgula e com todos os campos citados."""

    return dataframe.to_csv(
        index=False,
        sep=";",
        quoting=csv.QUOTE_ALL,
    ).encode("utf-8")


def uploaded_files_fingerprint(uploaded_files) -> str:
    """Calcula uma assinatura estável de nome, tamanho e conteúdo."""

    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(str(len(data)).encode("utf-8"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def make_arrow_friendly(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Padroniza tipos apenas para a exibição por Arrow/Streamlit."""

    if dataframe is None or dataframe.empty:
        return dataframe
    display = dataframe.copy()
    for column in DISPLAY_STRING_COLUMNS:
        if column in display.columns:
            display[column] = display[column].astype(str)
    for column in TEXT_COLUMNS:
        if column in display.columns:
            display[column] = display[column].fillna("").astype(str)
    return display


def venn_category_for_row(
    in_lattes: object,
    in_orcid: object,
    in_openalex: object,
) -> str | None:
    key = (bool(in_lattes), bool(in_orcid), bool(in_openalex))
    return {
        (True, False, False): "Lattes_Apenas",
        (False, True, False): "ORCID_Apenas",
        (False, False, True): "OpenAlex_Apenas",
        (True, True, False): "Lattes_ORCID_Apenas",
        (True, False, True): "Lattes_OpenAlex_Apenas",
        (False, True, True): "ORCID_OpenAlex_Apenas",
        (True, True, True): "Lattes_ORCID_OpenAlex",
    }.get(key)


def compute_venn_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
    counts = {
        "Lattes_Apenas": 0,
        "ORCID_Apenas": 0,
        "OpenAlex_Apenas": 0,
        "Lattes_ORCID_Apenas": 0,
        "Lattes_OpenAlex_Apenas": 0,
        "ORCID_OpenAlex_Apenas": 0,
        "Lattes_ORCID_OpenAlex": 0,
    }
    for row in rows:
        category = venn_category_for_row(
            row.get("Presente no Lattes", False),
            row.get("Presente no ORCID", False),
            row.get("Presente na OpenAlex", False),
        )
        if category:
            counts[category] += 1
    return counts, sum(counts.values())


def _sankey_counts(outcomes: list[FileProcessingResult]) -> dict[str, int]:
    return {
        "files_with_orcid": sum(outcome.has_orcid for outcome in outcomes),
        "files_without_orcid": sum(not outcome.has_orcid for outcome in outcomes),
        "files_with_orcid_with_openalex": sum(
            outcome.has_orcid and outcome.has_openalex for outcome in outcomes
        ),
        "files_with_orcid_without_openalex": sum(
            outcome.has_orcid and not outcome.has_openalex for outcome in outcomes
        ),
        "files_without_orcid_with_openalex": sum(
            not outcome.has_orcid and outcome.has_openalex for outcome in outcomes
        ),
        "files_without_orcid_without_openalex": sum(
            not outcome.has_orcid and not outcome.has_openalex for outcome in outcomes
        ),
    }


def _warning_file_row(outcome: FileProcessingResult) -> dict[str, Any]:
    return {
        "Arquivo": outcome.filename,
        "nomeLattes": outcome.researcher_name,
        "Qtd_DOIs_Lattes": outcome.lattes_doi_count,
        "Qtd_DOIs_encontrados_OpenAlex": len(outcome.openalex_dois),
        "Exemplos_DOI_OpenAlex": "; ".join(outcome.openalex_dois[:10]),
    }


def _warning_doi_rows(outcome: FileProcessingResult) -> list[dict[str, Any]]:
    by_doi = {row.get("DOI"): row for row in outcome.rows}
    rows: list[dict[str, Any]] = []
    for doi in outcome.openalex_dois:
        original = by_doi.get(doi, {})
        rows.append(
            {
                "Arquivo": outcome.filename,
                "nomeLattes": outcome.researcher_name,
                "DOI": doi,
                "Motivo": (
                    "OpenAlex encontrado por DOI, mas ORCID ausente no HTML do Lattes"
                ),
                "MatchPairs_OpenAlex": original.get("MatchPairs_OpenAlex", "-"),
                "IdsOpenAlex_Provaveis": original.get("IdsOpenAlex_Provaveis", "-"),
                "NomesOpenAlex_Provaveis": original.get("NomesOpenAlex_Provaveis", "-"),
                "ORCIDs_Provaveis": original.get("ORCIDs_Provaveis", "-"),
                "MetodoMatch": original.get("MetodoMatch", "-"),
                "MatchScoreMax": original.get("MatchScoreMax", 0),
            }
        )
    return rows
