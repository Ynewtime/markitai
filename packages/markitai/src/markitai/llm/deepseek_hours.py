"""DeepSeek's peak and off-peak hours, Chinese public holidays included.

DeepSeek bills Beijing time 09:00-12:00 and 14:00-18:00, Monday to Friday, at
its peak rates, and every other hour, weekends and Chinese public holidays
included, at the off-peak half
(https://api-docs.deepseek.com/quick_start/pricing). LiteLLM 1.104.2 knows the
hours and the weekends but not the holidays, so it charged a holiday's
working hours at the peak rates.

Whether a weekday is a public holiday is known only for the dates reviewed
against the State Council's holiday schedule: 2026-10-08 to 2026-12-31 hold
none (from the 2026 schedule). A weekday peak hour outside them could fall on
a holiday, so it is left unpriced, and so is a request sent in one period and
answered in another. markitai-rust prices DeepSeek by the same rule and the
same dates.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

#: Beijing dates reviewed against the State Council's holiday schedule; they
#: hold no public holiday
REVIEWED_DAYS = (date(2026, 10, 8), date(2026, 12, 31))
_BEIJING = timezone(timedelta(hours=8))
_PEAK_SECONDS = ((9 * 3600, 12 * 3600), (14 * 3600, 18 * 3600))


def is_peak(at: float) -> bool | None:
    """Whether DeepSeek bills Unix time *at* at its peak rates.

    ``None`` for a weekday peak hour on a date not reviewed for holidays.
    """
    local = datetime.fromtimestamp(int(at), UTC).astimezone(_BEIJING)
    second = local.hour * 3600 + local.minute * 60 + local.second
    working_hour = any(start <= second < end for start, end in _PEAK_SECONDS)
    if local.weekday() >= 5 or not working_hour:
        return False
    if REVIEWED_DAYS[0] <= local.date() <= REVIEWED_DAYS[1]:
        return True
    return None


def _count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(value)
    return value


def _detail(holder: Any, key: str) -> Any:
    return None if holder is None else getattr(holder, key, None)


def deepseek_cost(
    entry: dict[str, Any], usage: Any, sent: float | None, done: float
) -> float | None:
    """One DeepSeek answer's cost at the rates of its period, None if unknown.

    The usage must add up the way DeepSeek reports it: cache hits (under
    either of their names, which must agree) are part of the prompt, hits and
    misses make up the prompt, and reasoning is part of the completion.
    """
    if sent is None or sent > done:
        return None
    period = (is_peak(sent), is_peak(done))
    if period == (False, False):
        off_peak = entry.get("off_peak_pricing") or {}
    elif period == (True, True):
        off_peak = {}
    else:
        return None
    try:
        prompt = _count(getattr(usage, "prompt_tokens", None))
        completion = _count(getattr(usage, "completion_tokens", None))
        hits = [
            count
            for count in (
                _count(getattr(usage, "prompt_cache_hit_tokens", None)),
                _count(
                    _detail(
                        getattr(usage, "prompt_tokens_details", None), "cached_tokens"
                    )
                ),
            )
            if count is not None
        ]
        miss = _count(getattr(usage, "prompt_cache_miss_tokens", None))
        reasoning = _count(
            _detail(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens"
            )
        )
        total = _count(getattr(usage, "total_tokens", None))
    except ValueError:
        return None
    if prompt is None or completion is None or len(set(hits)) > 1:
        return None
    hit = hits[0] if hits else 0
    if (
        hit > prompt
        or (miss is not None and hit + miss != prompt)
        or (reasoning is not None and reasoning > completion)
        or (total is not None and total != prompt + completion)
    ):
        return None

    def rate(key: str) -> float | None:
        value = off_peak.get(key, entry.get(key))
        return float(value) if isinstance(value, int | float) else None

    input_rate = rate("input_cost_per_token")
    hit_rate = rate("cache_read_input_token_cost")
    output_rate = rate("output_cost_per_token")
    if input_rate is None or hit_rate is None or output_rate is None:
        return None
    return (prompt - hit) * input_rate + hit * hit_rate + completion * output_rate
