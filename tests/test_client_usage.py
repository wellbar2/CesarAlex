from typing import Any

from cesaralex.clients import OpenAlexClient


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.headers: dict[str, str] = {}
        self.payloads = list(payloads)

    def get(self, *_args, **_kwargs) -> FakeResponse:
        return FakeResponse(self.payloads.pop(0))


def test_client_counts_requests_and_accumulates_reported_cost() -> None:
    session = FakeSession(
        [
            {"meta": {"cost_usd": 0.0001}, "results": []},
            {"id": "https://openalex.org/W1"},
            {"rate_limit": {"daily_budget_usd": 1.0}},
        ]
    )
    client = OpenAlexClient(["test-key"], session=session)

    client.get_json("works", params={"filter": "x:y"})
    client.get_json("works/W1")
    rate_limit = client.get_rate_limit()

    stats = client.usage_stats()
    assert rate_limit is not None
    assert stats["http_requests"] == 3
    assert stats["list_requests"] == 1
    assert stats["singleton_requests"] == 1
    assert stats["rate_requests"] == 1
    assert stats["reported_cost_usd"] == 0.0001
