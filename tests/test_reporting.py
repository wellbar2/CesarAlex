from cesaralex.models import FileProcessingResult
from cesaralex.reporting import build_application_results


def test_sankey_and_warning_include_openalex_found_without_orcid() -> None:
    outcome = FileProcessingResult(
        filename="pesquisador.html",
        researcher_name="Pesquisador Teste",
        orcid_url=None,
        rows=[
            {
                "Arquivo": "pesquisador.html",
                "nomeLattes": "Pesquisador Teste",
                "DOI": "10.1000/example",
                "Presente no Lattes": True,
                "Presente no ORCID": False,
                "Presente na OpenAlex": True,
                "IdsOpenAlex_Provaveis": "A1",
                "NomesOpenAlex_Provaveis": "Pesquisador Teste",
            }
        ],
        lattes_doi_count=1,
        openalex_dois=["10.1000/example"],
        openalex_detected=True,
    )

    results = build_application_results(
        [outcome],
        check_openalex_by_doi=True,
    )

    assert results["sankey"]["files_without_orcid_with_openalex"] == 1
    assert results["warn_files_df"].iloc[0]["Qtd_DOIs_encontrados_OpenAlex"] == 1
    assert results["warn_rows_df"].iloc[0]["IdsOpenAlex_Provaveis"] == "A1"
