from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_exception() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()

    assert not app.exception
    assert app.title[0].value == "Integração de Currículo Lattes, ORCID e OpenAlex"
    assert "Atualizar saldos" in [button.label for button in app.button]
    assert "Processar currículos" in [button.label for button in app.button]
    assert len(app.get("progress")) == 3
