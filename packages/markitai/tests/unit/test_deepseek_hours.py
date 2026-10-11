"""DeepSeek's peak hours: Beijing time, weekends and public holidays off-peak.

The same rule and reviewed dates as markitai-rust's catalog
(crates/markitai-core/src/pricing/catalog.rs, deepseek_peak).
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from markitai.llm.deepseek_hours import deepseek_cost, is_peak

importlib.import_module("markitai.llm")  # bundled cost map, then litellm
litellm = importlib.import_module("litellm")


def _at(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


MONDAY = "2026-10-12"  # a reviewed working day; Beijing is UTC+8


@pytest.mark.parametrize(
    ("utc", "peak"),
    [
        (f"{MONDAY}T00:59:59", False),  # 08:59:59 Beijing
        (f"{MONDAY}T01:00:00", True),  # 09:00
        (f"{MONDAY}T03:59:59", True),
        (f"{MONDAY}T04:00:00", False),  # 12:00, lunch
        (f"{MONDAY}T06:00:00", True),  # 14:00
        (f"{MONDAY}T09:59:59", True),
        (f"{MONDAY}T10:00:00", False),  # 18:00
        ("2026-10-17T02:00:00", False),  # Saturday
        ("2026-10-18T02:00:00", False),  # Sunday
        # Beijing's Saturday starts at 16:00 UTC on Friday
        ("2026-10-16T16:30:00", False),
        ("2026-12-31T02:00:00", True),  # last reviewed day
        # Not reviewed for public holidays: National Day week, and 2027
        ("2026-10-05T02:00:00", None),
        ("2027-01-04T02:00:00", None),
        ("2027-01-04T05:00:00", False),  # 13:00: off-peak whatever the date
    ],
)
def test_peak_hours(utc: str, peak: bool | None) -> None:
    assert is_peak(_at(utc)) is peak


_ENTRY = {
    "input_cost_per_token": 3e-07,
    "output_cost_per_token": 1.2e-06,
    "cache_read_input_token_cost": 6e-09,
    "off_peak_pricing": {
        "input_cost_per_token": 1.5e-07,
        "output_cost_per_token": 6e-07,
        "cache_read_input_token_cost": 3e-09,
    },
}


def _usage(**extra: object) -> object:
    fields = {
        "prompt_tokens": 10_000,
        "completion_tokens": 1_000,
        "total_tokens": 11_000,
        "prompt_cache_hit_tokens": 4_000,
        "prompt_cache_miss_tokens": 6_000,
        "prompt_tokens_details": {"cached_tokens": 4_000},
        **extra,
    }
    return litellm.Usage(**fields)


PEAK_COST = 6_000 * 3e-07 + 4_000 * 6e-09 + 1_000 * 1.2e-06
OFF_PEAK_COST = 6_000 * 1.5e-07 + 4_000 * 3e-09 + 1_000 * 6e-07


@pytest.mark.parametrize(
    ("sent", "done", "cost"),
    [
        (f"{MONDAY}T02:00:00", f"{MONDAY}T02:00:30", PEAK_COST),
        (f"{MONDAY}T05:00:00", f"{MONDAY}T05:00:30", OFF_PEAK_COST),
        ("2026-10-17T02:00:00", "2026-10-17T02:00:30", OFF_PEAK_COST),
        # A holiday's working hours: not known to be peak, so not priced
        ("2026-10-05T02:00:00", "2026-10-05T02:00:30", None),
        # Sent off-peak, answered at peak
        (f"{MONDAY}T00:59:50", f"{MONDAY}T01:00:20", None),
    ],
)
def test_cost_by_period(sent: str, done: str, cost: float | None) -> None:
    result = deepseek_cost(_ENTRY, _usage(), _at(sent), _at(done))
    if cost is None:
        assert result is None
    else:
        assert result == pytest.approx(cost)


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt_cache_miss_tokens": 5_000},  # hits + misses != prompt
        {"total_tokens": 12_000},
        {"completion_tokens_details": {"reasoning_tokens": 2_000}},
    ],
)
def test_contradictory_usage_is_not_priced(extra: dict) -> None:
    sent = _at(f"{MONDAY}T05:00:00")
    assert deepseek_cost(_ENTRY, _usage(**extra), sent, sent + 30) is None


def test_hit_counts_under_both_names_must_agree() -> None:
    """litellm.Usage copies prompt_cache_hit_tokens over cached_tokens, so
    only a raw usage object can disagree; it is not priced either."""
    from types import SimpleNamespace

    usage = SimpleNamespace(
        prompt_tokens=10_000,
        completion_tokens=1_000,
        prompt_cache_hit_tokens=4_000,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3_000),
    )
    sent = _at(f"{MONDAY}T05:00:00")
    assert deepseek_cost(_ENTRY, usage, sent, sent + 30) is None


def test_get_response_cost_uses_the_send_time() -> None:
    from markitai.llm.models import get_response_cost, stamp_sent_at

    def answer(sent: float) -> object:
        response = litellm.ModelResponse(model="deepseek-flash", usage=_usage())
        response._hidden_params = {"custom_llm_provider": "deepseek"}
        stamp_sent_at(response, sent)
        return response

    holiday = _at("2026-10-05T02:00:00")
    working_day = _at(f"{MONDAY}T02:00:00")
    with patch("time.time", return_value=holiday + 30):
        # LiteLLM alone charged this at the peak rates
        assert get_response_cost(answer(holiday)) is None
    with patch("time.time", return_value=working_day + 30):
        assert get_response_cost(answer(working_day)) == pytest.approx(PEAK_COST)


def test_without_a_stamp_the_providers_created_time_is_used() -> None:
    from markitai.llm.models import get_response_cost

    sent = _at(f"{MONDAY}T05:00:00")
    response = litellm.ModelResponse(
        model="deepseek-flash", usage=_usage(), created=int(sent)
    )
    response._hidden_params = {"custom_llm_provider": "deepseek"}
    with patch("time.time", return_value=sent + 30):
        assert get_response_cost(response) == pytest.approx(OFF_PEAK_COST)
