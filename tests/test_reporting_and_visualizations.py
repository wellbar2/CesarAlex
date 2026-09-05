from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from cesaralex.reporting import (
    compute_venn_counts,
    dataframe_to_csv_bytes,
    make_arrow_friendly,
    uploaded_files_fingerprint,
)
from cesaralex.visualizations import create_sankey, create_venn


class UploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.data = data

    def getvalue(self) -> bytes:
        return self.data


def test_csv_format_and_uploaded_file_fingerprint_are_stable() -> None:
    frame = pd.DataFrame([{"DOI": "10.1/a", "Título": "A; B"}])
    csv_text = dataframe_to_csv_bytes(frame).decode("utf-8")
    files = [UploadedFile("a.html", b"content")]

    assert csv_text.startswith('"DOI";"Título"')
    assert '"A; B"' in csv_text
    assert uploaded_files_fingerprint(files) == uploaded_files_fingerprint(files)
    assert uploaded_files_fingerprint(files) != uploaded_files_fingerprint(
        [UploadedFile("b.html", b"content")]
    )


def test_arrow_conversion_and_venn_counts_cover_all_regions() -> None:
    frame = pd.DataFrame([{"Ano": 2025, "DOI": None, "Presente na OpenAlex": True}])
    display = make_arrow_friendly(frame)
    rows = [
        {
            "Presente no Lattes": bool(mask & 4),
            "Presente no ORCID": bool(mask & 2),
            "Presente na OpenAlex": bool(mask & 1),
        }
        for mask in range(1, 8)
    ]
    counts, total = compute_venn_counts(rows)

    assert display.loc[0, "Ano"] == "2025"
    assert display.loc[0, "DOI"] == ""
    assert display.loc[0, "Presente na OpenAlex"] == "True"
    assert total == 7
    assert set(counts.values()) == {1}


def test_visualizations_are_created_from_coverage_counts() -> None:
    sankey_counts = {
        "files_with_orcid": 2,
        "files_without_orcid": 1,
        "files_with_orcid_with_openalex": 1,
        "files_with_orcid_without_openalex": 1,
        "files_without_orcid_with_openalex": 1,
        "files_without_orcid_without_openalex": 0,
    }
    venn_counts = {
        "Lattes_Apenas": 1,
        "ORCID_Apenas": 1,
        "OpenAlex_Apenas": 1,
        "Lattes_ORCID_Apenas": 1,
        "Lattes_OpenAlex_Apenas": 1,
        "ORCID_OpenAlex_Apenas": 1,
        "Lattes_ORCID_OpenAlex": 1,
    }

    sankey = create_sankey(sankey_counts)
    venn = create_venn(venn_counts, total=7, title="Coverage")

    assert tuple(sankey.data[0]["link"]["value"]) == (2, 1, 1, 1, 1, 0)
    assert venn.axes[0].get_title() == "Coverage"
    plt.close(venn)
