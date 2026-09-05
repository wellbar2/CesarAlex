from pathlib import Path

import pandas as pd

from examples.offline_core_example import main


def test_offline_example_writes_expected_report(tmp_path: Path) -> None:
    output_path = main(tmp_path / "example_report.csv")

    report = pd.read_csv(output_path, sep=";")
    assert len(report) == 2
    assert report["Presente no Lattes"].all()
    assert report["Presente na OpenAlex"].all()
    assert report["Presente no ORCID"].sum() == 1
