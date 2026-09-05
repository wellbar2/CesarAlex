"""Clientes HTTP e acesso tipado às APIs ORCID e OpenAlex."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import requests

from .config import DEFAULT_SETTINGS, Settings
from .models import Author, DoiResolution, LookupSource, Publication
from .normalization import (
    extract_orcid_id,
    normalize_doi,
    normalize_openalex_author_id,
)

MessageCallback = Callable[[str], None]
UsageCallback = Callable[[dict[str, int | float]], None]


def _ignore_message(_message: str) -> None:
    return None


def _ignore_usage(_stats: dict[str, int | float]) -> None:
    return None


def parse_openalex_api_keys(keys_text: str) -> list[str]:
    """Aceita uma chave ou várias chaves separadas por ponto e vírgula."""

    return [key.strip() for key in (keys_text or "").split(";") if key.strip()]


class OpenAlexClient:
    """Cliente HTTP com rotação de chaves e retry exponencial."""

    base_url = "https://api.openalex.org"

    def __init__(
        self,
        api_keys: list[str],
        *,
        settings: Settings = DEFAULT_SETTINGS,
        session: requests.Session | None = None,
        on_warning: MessageCallback = _ignore_message,
        on_usage_change: UsageCallback = _ignore_usage,
    ) -> None:
        self.api_keys = [key.strip() for key in api_keys if key.strip()]
        if not self.api_keys:
            raise ValueError("Nenhuma chave OpenAlex foi informada.")
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "CesarAlex/1.0"})
        self.on_warning = on_warning
        self.on_usage_change = on_usage_change
        self._key_index = 0
        self.last_error = ""
        self._http_requests = 0
        self._successful_requests = 0
        self._retry_requests = 0
        self._list_requests = 0
        self._singleton_requests = 0
        self._rate_requests = 0
        self._reported_cost_usd = 0.0

    @property
    def current_api_key(self) -> str:
        return self.api_keys[self._key_index]

    @property
    def key_position_label(self) -> str:
        return f"{self._key_index + 1}/{len(self.api_keys)}"

    def get_json(
        self,
        endpoint_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any] | None:
        url = (
            endpoint_or_url
            if endpoint_or_url.startswith(("http://", "https://"))
            else f"{self.base_url}/{endpoint_or_url.lstrip('/')}"
        )
        timeout = timeout or self.settings.openalex_doi_timeout
        last_exception: requests.RequestException | None = None

        for attempt in range(1, self.settings.openalex_max_retries + 1):
            request_params = dict(params or {})
            request_params["api_key"] = self.current_api_key
            self._record_request(url, is_retry=attempt > 1)
            try:
                response = self.session.get(url, params=request_params, timeout=timeout)
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        self.last_error = "A OpenAlex retornou JSON inválido."
                        return None
                    if not isinstance(data, dict):
                        self.last_error = "A OpenAlex retornou um formato inesperado."
                        return None
                    self.last_error = ""
                    self._successful_requests += 1
                    meta = data.get("meta") or {}
                    self._reported_cost_usd += _as_non_negative_float(
                        meta.get("cost_usd") if isinstance(meta, dict) else None
                    )
                    self._notify_usage_change()
                    return data
                if response.status_code == 429:
                    if self._advance_key():
                        self.on_warning(
                            "Limite da chave OpenAlex atingido. "
                            "Alternando para a próxima chave "
                            f"({self.key_position_label})."
                        )
                        time.sleep(1.0)
                        continue
                    self.last_error = (
                        "OpenAlex retornou 429 e não há outra chave disponível. "
                        f"Tentativa {attempt}/{self.settings.openalex_max_retries}."
                    )
                    time.sleep(self._backoff(attempt))
                    continue
                if 500 <= response.status_code < 600:
                    self.last_error = (
                        f"Erro transitório OpenAlex {response.status_code}. "
                        f"Tentativa {attempt}/{self.settings.openalex_max_retries}."
                    )
                    time.sleep(self._backoff(attempt))
                    continue
                self.last_error = (
                    f"Erro OpenAlex HTTP {response.status_code}: {response.text[:500]}"
                )
                return None
            except requests.RequestException as exc:
                last_exception = exc
                self.last_error = (
                    f"Falha de rede/requisição na OpenAlex: {exc}. "
                    f"Tentativa {attempt}/{self.settings.openalex_max_retries}."
                )
                time.sleep(self._backoff(attempt))

        if last_exception:
            self.last_error = f"Falha final na OpenAlex: {last_exception}"
        elif not self.last_error:
            self.last_error = "Falha final na OpenAlex após todas as tentativas."
        return None

    def get_rate_limit(self) -> dict[str, Any] | None:
        """Consulta os saldos associados à chave atualmente ativa."""

        data = self.get_json("rate-limit")
        if data is None:
            return None
        if not isinstance(data.get("rate_limit"), dict):
            self.last_error = (
                "A OpenAlex não retornou informações de saldo em `rate_limit`."
            )
            return None
        return data

    def usage_stats(self) -> dict[str, int | float]:
        return {
            "http_requests": self._http_requests,
            "successful_requests": self._successful_requests,
            "retry_requests": self._retry_requests,
            "list_requests": self._list_requests,
            "singleton_requests": self._singleton_requests,
            "rate_requests": self._rate_requests,
            "reported_cost_usd": self._reported_cost_usd,
        }

    def _record_request(self, url: str, *, is_retry: bool) -> None:
        self._http_requests += 1
        if is_retry:
            self._retry_requests += 1
        path = urlparse(url).path.rstrip("/")
        if path.endswith("/rate-limit"):
            self._rate_requests += 1
        elif path.endswith("/works"):
            self._list_requests += 1
        else:
            self._singleton_requests += 1
        self._notify_usage_change()

    def _notify_usage_change(self) -> None:
        self.on_usage_change(self.usage_stats())

    def _advance_key(self) -> bool:
        if self._key_index >= len(self.api_keys) - 1:
            return False
        self._key_index += 1
        return True

    def _backoff(self, attempt: int) -> float:
        return self.settings.openalex_backoff_base ** (attempt - 1)


def _as_non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


class OrcidService:
    """Recupera as obras públicas associadas a um ORCID."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        on_error: MessageCallback = _ignore_message,
    ) -> None:
        self.session = session or requests.Session()
        self.on_error = on_error

    def publications_by_orcid(self, orcid_url: str) -> list[Publication]:
        orcid_id = extract_orcid_id(orcid_url)
        if not orcid_id:
            return []
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
        try:
            response = self.session.get(
                url,
                headers={"Accept": "application/json"},
                timeout=30,
            )
        except requests.RequestException:
            self.on_error(f"Falha ao acessar ORCID API para {orcid_id}.")
            return []
        if response.status_code != 200:
            self.on_error(
                f"Erro ORCID API para {orcid_id}. Código: {response.status_code}"
            )
            return []
        return _parse_orcid_publications(response.json())


class OpenAlexService:
    """Camada de domínio OpenAlex com caches válidos durante o processamento."""

    def __init__(
        self,
        client: OpenAlexClient,
        *,
        settings: Settings = DEFAULT_SETTINGS,
        on_error: MessageCallback = _ignore_message,
    ) -> None:
        self.client = client
        self.settings = settings
        self.on_error = on_error
        self._doi_cache: dict[str, DoiResolution] = {}
        self._author_cache: dict[str, tuple[str | None, str | None]] = {}

    def publications_by_orcid(self, orcid_url: str) -> list[Publication]:
        orcid_id = extract_orcid_id(orcid_url)
        if not orcid_id:
            return []
        return self._paginated_publications(
            filter_value=f"authorships.author.orcid:{orcid_id}",
            context=f"ORCID {orcid_id}",
        )

    def publications_by_author(self, author_id: str) -> list[Publication]:
        author_id = normalize_openalex_author_id(author_id)
        if not author_id:
            return []
        return self._paginated_publications(
            filter_value=f"authorships.author.id:{author_id}",
            context=f"author_id {author_id}",
        )

    def resolve_doi(self, doi: str) -> DoiResolution:
        doi = normalize_doi(doi) or ""
        if not doi:
            return DoiResolution.not_found()
        if doi in self._doi_cache:
            cached = self._doi_cache[doi]
            return DoiResolution(
                found=cached.found,
                source=LookupSource.DOI_CACHE,
                publication=cached.publication,
                made_request=False,
            )

        data = self.client.get_json(
            f"works/http://dx.doi.org/{doi}",
            timeout=self.settings.openalex_doi_timeout,
        )
        if data is None:
            resolution = DoiResolution.not_found(made_request=True)
        else:
            resolution = DoiResolution(
                found=True,
                source=LookupSource.DOI_API,
                publication=_parse_openalex_publication(data),
                made_request=True,
            )
        self._doi_cache[doi] = resolution
        return resolution

    def author_orcid(self, author_id: str) -> str | None:
        author_id = normalize_openalex_author_id(author_id)
        if not author_id:
            return None
        if author_id not in self._author_cache:
            data = self.client.get_json(
                f"authors/{author_id}",
                timeout=self.settings.openalex_doi_timeout,
            )
            self._author_cache[author_id] = (
                data.get("orcid") if data else None,
                data.get("display_name") if data else None,
            )
        return self._author_cache[author_id][0]

    def _paginated_publications(
        self,
        *,
        filter_value: str,
        context: str,
    ) -> list[Publication]:
        publications: list[Publication] = []
        cursor = "*"
        for _page in range(self.settings.openalex_max_pages):
            data = self.client.get_json(
                "works",
                params={
                    "filter": filter_value,
                    "per_page": self.settings.openalex_per_page,
                    "cursor": cursor,
                },
                timeout=30,
            )
            if data is None:
                self.on_error(f"Erro OpenAlex para {context}. {self.client.last_error}")
                break
            results = data.get("results", [])
            if not results:
                break
            publications.extend(_parse_openalex_publication(work) for work in results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return publications


def _parse_orcid_publications(data: dict[str, Any]) -> list[Publication]:
    publications: list[Publication] = []
    for group in data.get("group", []):
        for summary in group.get("work-summary", []):
            title = ((summary.get("title") or {}).get("title") or {}).get(
                "value"
            ) or "Sem título"
            year = ((summary.get("publication-date") or {}).get("year") or {}).get(
                "value"
            ) or "Sem data"
            doi = None
            for external_id in (summary.get("external-ids") or {}).get(
                "external-id", []
            ):
                if str(external_id.get("external-id-type", "")).lower() == "doi":
                    doi = normalize_doi(external_id.get("external-id-value"))
                    break
            publications.append(Publication(doi=doi, title=title, year=year))
    return publications


def _parse_openalex_publication(work: dict[str, Any]) -> Publication:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    authors: list[Author] = []
    for authorship in work.get("authorships", []):
        author_data = authorship.get("author") or {}
        author_id = normalize_openalex_author_id(author_data.get("id"))
        author_name = str(author_data.get("display_name") or "").strip()
        if author_id and author_name:
            authors.append(
                Author(
                    openalex_id=author_id,
                    name=author_name,
                    orcid=author_data.get("orcid"),
                )
            )
    return Publication(
        doi=normalize_doi(work.get("doi")),
        title=work.get("display_name") or "Sem título",
        year=work.get("publication_year") or "Sem data",
        issn_l=source.get("issn_l"),
        authors=tuple(authors),
    )
