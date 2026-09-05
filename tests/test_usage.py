from cesaralex.usage import (
    aggregate_balances,
    balance_from_rate_limit_payload,
)


def test_rate_limit_payload_is_parsed_and_balances_are_aggregated() -> None:
    first = balance_from_rate_limit_payload(
        {
            "rate_limit": {
                "daily_budget_usd": 1.0,
                "daily_remaining_usd": 0.75,
                "daily_used_usd": 0.25,
                "prepaid_balance_usd": 5.0,
                "prepaid_remaining_usd": 4.5,
            }
        }
    )
    second = balance_from_rate_limit_payload(
        {
            "rate_limit": {
                "daily_budget_usd": 2.0,
                "daily_remaining_usd": 1.5,
                "daily_used_usd": 0.5,
                "prepaid_balance_usd": 3.0,
                "prepaid_remaining_usd": 3.0,
            }
        }
    )

    total = aggregate_balances([first, second])

    assert total is not None
    assert total.key_count == 2
    assert total.daily_budget_usd == 3.0
    assert total.daily_remaining_usd == 2.25
    assert total.daily_used_usd == 0.75
    assert total.prepaid_remaining_usd == 7.5
    assert total.prepaid_reference_usd == 8.0


def test_cost_estimate_uses_daily_balance_before_prepaid_balance() -> None:
    balance = balance_from_rate_limit_payload(
        {
            "rate_limit": {
                "daily_budget_usd": 0.002,
                "daily_remaining_usd": 0.001,
                "daily_used_usd": 0.001,
                "prepaid_balance_usd": 1.0,
                "prepaid_remaining_usd": 1.0,
            }
        }
    )

    estimated = balance.estimate_after_cost(0.0015)

    assert estimated.daily_remaining_usd == 0.0
    assert estimated.daily_used_usd == 0.002
    assert estimated.prepaid_remaining_usd == 0.9995


def test_daily_used_is_derived_when_endpoint_omits_it() -> None:
    balance = balance_from_rate_limit_payload(
        {
            "rate_limit": {
                "daily_budget_usd": 1.0,
                "daily_remaining_usd": 0.65,
            }
        }
    )

    assert balance.daily_used_usd == 0.35
