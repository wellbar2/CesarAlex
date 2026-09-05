from cesaralex.lattes import parse_lattes_html
from cesaralex.normalization import normalize_doi


def test_normalize_doi_accepts_common_prefixes() -> None:
    assert normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert normalize_doi("doi:10.1234/ABC") == "10.1234/abc"
    assert normalize_doi("Sem DOI") is None


def test_parse_lattes_extracts_name_orcid_title_year_and_doi() -> None:
    html = """
    <html><body>
      <h2 class="nome">Ana Maria da Silva</h2>
      <a href="https://orcid.org/0000-0000-0000-0001">ORCID</a>
      <span class="transform">
        Autores . Título visível. Revista, v. 1, 2025.
        <span class="citado" cvuri="https://x.test/?titulo=T%C3%ADtulo%20exato"></span>
        <a class="icone-producao icone-doi" href="https://doi.org/10.1234/ABC">DOI</a>
      </span>
    </body></html>
    """

    curriculum = parse_lattes_html(html)

    assert curriculum.researcher_name == "Ana Maria da Silva"
    assert curriculum.orcid_url == "https://orcid.org/0000-0000-0000-0001"
    assert curriculum.articles["10.1234/abc"].title == "Título exato"
    assert curriculum.articles["10.1234/abc"].year == "2025"
