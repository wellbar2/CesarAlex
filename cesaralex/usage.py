"""Cálculo de saldos e custos exibidos no menu lateral."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any


def _as_non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True, slots=True)
class OpenAlexBalance:
    """Saldo agregado de uma ou mais chaves OpenAlex."""

    daily_budget_usd: float = 0.0
    daily_remaining_usd: float = 0.0
    daily_used_usd: float = 0.0
    prepaid_remaining_usd: float = 0.0
    prepaid_reference_usd: float = 0.0
    key_count: int = 0

    @property
    def daily_remaining_fraction(self) -> float:
        if self.daily_budget_usd <= 0:
            return 0.0
        return min(1.0, self.daily_remaining_usd / self.daily_budget_usd)

    @property
    def prepaid_remaining_fraction(self) -> float:
        if self.prepaid_reference_usd <= 0:
            return 0.0
        return min(1.0, self.prepaid_remaining_usd / self.prepaid_reference_usd)

    def estimate_after_cost(self, cost_usd: float) -> OpenAlexBalance:
        """Atualiza visualmente os saldos entre duas leituras de `/rate-limit`."""

        cost_usd = _as_non_negative_float(cost_usd)
        daily_charge = min(cost_usd, self.daily_remaining_usd)
        prepaid_charge = max(0.0, cost_usd - daily_charge)
        return replace(
            self,
            daily_remaining_usd=max(0.0, self.daily_remaining_usd - daily_charge),
            daily_used_usd=self.daily_used_usd + daily_charge,
            prepaid_remaining_usd=max(
                0.0,
                self.prepaid_remaining_usd - prepaid_charge,
            ),
        )


def balance_from_rate_limit_payload(payload: dict[str, Any]) -> OpenAlexBalance:
    rate_limit = payload.get("rate_limit") or {}
    daily_budget = _as_non_negative_float(rate_limit.get("daily_budget_usd"))
    daily_remaining = _as_non_negative_float(rate_limit.get("daily_remaining_usd"))
    daily_used_value = rate_limit.get("daily_used_usd")
    daily_used = (
        _as_non_negative_float(daily_used_value)
        if daily_used_value is not None
        else max(0.0, daily_budget - daily_remaining)
    )
    prepaid_balance = _as_non_negative_float(rate_limit.get("prepaid_balance_usd"))
    prepaid_remaining = _as_non_negative_float(rate_limit.get("prepaid_remaining_usd"))
    return OpenAlexBalance(
        daily_budget_usd=daily_budget,
        daily_remaining_usd=daily_remaining,
        daily_used_usd=daily_used,
        prepaid_remaining_usd=prepaid_remaining,
        prepaid_reference_usd=max(prepaid_balance, prepaid_remaining),
        key_count=1,
    )


def aggregate_balances(balances: list[OpenAlexBalance]) -> OpenAlexBalance | None:
    if not balances:
        return None
    return OpenAlexBalance(
        daily_budget_usd=sum(balance.daily_budget_usd for balance in balances),
        daily_remaining_usd=sum(balance.daily_remaining_usd for balance in balances),
        daily_used_usd=sum(balance.daily_used_usd for balance in balances),
        prepaid_remaining_usd=sum(
            balance.prepaid_remaining_usd for balance in balances
        ),
        prepaid_reference_usd=sum(
            balance.prepaid_reference_usd for balance in balances
        ),
        key_count=sum(balance.key_count for balance in balances),
    )


def format_usd(value: float) -> str:
    text = f"{_as_non_negative_float(value):,.4f}"
    localized = text.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"US$ {localized}"


def api_keys_fingerprint(api_keys: list[str]) -> str:
    """Identifica a combinação de chaves sem guardá-las na sessão."""

    digest = hashlib.sha256()
    for api_key in api_keys:
        digest.update(api_key.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
