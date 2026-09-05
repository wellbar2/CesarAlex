from cesaralex.matching import find_author_matches, matches_to_report
from cesaralex.models import Author


def test_match_report_uses_openalex_name_and_id() -> None:
    matches = find_author_matches(
        "Wellington Barbosa Rodrigues",
        [Author("A123", "Wellington Barbosa Rodrigues")],
    )

    report = matches_to_report(matches)

    assert report["IdsOpenAlex_Provaveis"] == "A123"
    assert report["NomesOpenAlex_Provaveis"] == "Wellington Barbosa Rodrigues"
    assert report["MatchScoreMax"] == 3
