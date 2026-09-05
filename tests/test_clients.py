from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import requests

from cesaralex.clients import OpenAlexClient, OpenAlexService, OrcidService
from cesaralex.config import DEFAULT_SETTINGS
from cesaralex.models import LookupSource


class Response:
    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        *,
        text: str = "",
        invalid_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.text = text
        self.invalid_json = invalid_json

    def json(self) -> Any:
        if self.invalid_json:
            raise ValueError("invalid JSON")
        return self.payload


class QueueSession:
    def __init__(self, items: list[Response | requests.RequestException]) -> None:
        self.headers: dict[str, str] = {}
        self.items = list(items)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> Response:
        self.calls.append((url, kwargs))
        item = self.items.pop(0)
        if isinstance(item, requests.RequestException):
            raise item
        return item


class StubClient:
    def __init__(self, payloads: list[dict[str, Any] | None]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.last_error = "simulated error"

    def get_json(self, endpoint: str, **kwargs: Any) -> dict[str, Any] | None:
        self.calls.append((endpoint, kwargs))
        return self.payloads.pop(0)


def work_payload(doi: str = "10.1000/example") -> dict[str, Any]:
    return {
        "doi": f"https://doi.org/{doi}",
        "display_name": "Example article",
        "publication_year": 2025,
        "primary_location": {"source": {"issn_l": "1234-5678"}},
        "authorships": [
            {
                "author": {
                    "id": "https://openalex.org/A123",
                    "display_name": "Ana Example",
                    "orcid": "https://orcid.org/0000-0000-0000-0000",
                }
            }
        ],
    }


def test_openalex_client_requires_a_key() -> None:
    with pytest.raises(ValueError, match="Nenhuma chave"):
        OpenAlexClient([])


def test_openalex_client_rotates_key_after_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr("cesaralex.clients.time.sleep", lambda _seconds: None)
    session = QueueSession([Response(429), Response(200, {"results": []})])
    warnings: list[str] = []
    settings = replace(DEFAULT_SETTINGS, openalex_max_retries=2)
    client = OpenAlexClient(
        ["first", "second"],
        session=session,
        settings=settings,
        on_warning=warnings.append,
    )

    assert client.get_json("works") == {"results": []}
    assert session.calls[0][1]["params"]["api_key"] == "first"
    assert session.calls[1][1]["params"]["api_key"] == "second"
    assert client.usage_stats()["retry_requests"] == 1
    assert warnings and "Alternando" in warnings[0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (Response(200, invalid_json=True), "JSON inválido"),
        (Response(200, []), "formato inesperado"),
        (Response(400, text="bad request"), "HTTP 400"),
    ],
)
def test_openalex_client_reports_non_successful_payloads(
    response: Response,
    message: str,
) -> None:
    client = OpenAlexClient(["key"], session=QueueSession([response]))

    assert client.get_json("works/W1") is None
    assert message in client.last_error


def test_openalex_client_reports_final_network_failure(monkeypatch) -> None:
    monkeypatch.setattr("cesaralex.clients.time.sleep", lambda _seconds: None)
    settings = replace(DEFAULT_SETTINGS, openalex_max_retries=1)
    session = QueueSession([requests.ConnectionError("offline")])
    client = OpenAlexClient(["key"], session=session, settings=settings)

    assert client.get_json("works/W1") is None
    assert "Falha final" in client.last_error


def test_openalex_service_paginates_and_parses_publications() -> None:
    client = StubClient(
        [
            {"results": [work_payload()], "meta": {"next_cursor": "next"}},
            {"results": [], "meta": {}},
        ]
    )
    service = OpenAlexService(client)  # type: ignore[arg-type]

    publications = service.publications_by_orcid(
        "https://orcid.org/0000-0000-0000-0000"
    )

    assert len(publications) == 1
    assert publications[0].doi == "10.1000/example"
    assert publications[0].issn_l == "1234-5678"
    assert publications[0].authors[0].openalex_id == "A123"
    assert len(client.calls) == 2
    assert "authorships.author.orcid" in client.calls[0][1]["params"]["filter"]


def test_openalex_service_caches_doi_and_author_queries() -> None:
    client = StubClient(
        [
            work_payload(),
            {
                "orcid": "https://orcid.org/0000-0000-0000-0000",
                "display_name": "Ana Example",
            },
        ]
    )
    service = OpenAlexService(client)  # type: ignore[arg-type]

    first = service.resolve_doi("https://doi.org/10.1000/EXAMPLE")
    second = service.resolve_doi("10.1000/example")
    first_orcid = service.author_orcid("https://openalex.org/A123")
    second_orcid = service.author_orcid("A123")

    assert first.source == LookupSource.DOI_API
    assert first.made_request is True
    assert second.source == LookupSource.DOI_CACHE
    assert second.made_request is False
    assert first_orcid == second_orcid
    assert len(client.calls) == 2


def test_openalex_service_reports_pagination_failure() -> None:
    client = StubClient([None])
    errors: list[str] = []
    service = OpenAlexService(client, on_error=errors.append)  # type: ignore[arg-type]

    assert service.publications_by_author("A123") == []
    assert errors and "simulated error" in errors[0]


def test_orcid_service_parses_public_works() -> None:
    payload = {
        "group": [
            {
                "work-summary": [
                    {
                        "title": {"title": {"value": "ORCID article"}},
                        "publication-date": {"year": {"value": "2024"}},
                        "external-ids": {
                            "external-id": [
                                {
                                    "external-id-type": "doi",
                                    "external-id-value": "10.1000/ORCID",
                                }
                            ]
                        },
                    }
                ]
            }
        ]
    }
    session = QueueSession([Response(200, payload)])
    service = OrcidService(session=session)

    publications = service.publications_by_orcid(
        "https://orcid.org/0000-0000-0000-0000"
    )

    assert publications[0].doi == "10.1000/orcid"
    assert publications[0].title == "ORCID article"
    assert session.calls[0][1]["headers"] == {"Accept": "application/json"}


def test_orcid_service_reports_http_and_network_errors() -> None:
    errors: list[str] = []
    http_service = OrcidService(
        session=QueueSession([Response(503)]),
        on_error=errors.append,
    )
    network_service = OrcidService(
        session=QueueSession([requests.ConnectionError("offline")]),
        on_error=errors.append,
    )

    assert http_service.publications_by_orcid("0000-0000-0000-0000") == []
    assert network_service.publications_by_orcid("0000-0000-0000-0001") == []
    assert any("Código: 503" in message for message in errors)
    assert any("Falha ao acessar" in message for message in errors)
