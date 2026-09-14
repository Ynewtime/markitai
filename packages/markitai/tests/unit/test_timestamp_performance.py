"""Timestamp optimization must retain local offsets and clock precision."""

import os
import time
from datetime import UTC, datetime

import pytest

from markitai.runs import report
from markitai.utils import clock, frontmatter


@pytest.mark.parametrize("module", [clock, frontmatter, report])
def test_timestamp_reads_an_aware_clock(module, monkeypatch):
    """Avoid the costly conversion from an ambiguous naive local instant."""
    instant = datetime(2024, 11, 3, 6, 30, 0, 123456, tzinfo=UTC)

    class AwareClock:
        @staticmethod
        def now(tz=None):
            assert tz is not None, "Read an aware clock before resolving local time"
            return instant.astimezone(tz)

    monkeypatch.setattr(module, "datetime", AwareClock)
    if module is report:
        actual = report.build_report_shell(log_file=None)["generated_at"]
        expected = instant.astimezone().isoformat()
    else:
        actual = (
            clock.now_iso() if module is clock else frontmatter.frontmatter_timestamp()
        )
        expected = instant.astimezone().isoformat(timespec="milliseconds")
    assert actual == expected


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="Requires POSIX TZ switching")
@pytest.mark.parametrize(
    "zone",
    [
        "UTC",
        "Asia/Shanghai",
        "America/New_York",
        "Australia/Lord_Howe",
        "Asia/Kathmandu",
    ],
)
def test_timestamp_matches_previous_output_across_timezone_changes(zone, monkeypatch):
    previous = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        # Spring/fall DST boundaries, both occurrences of a repeated local
        # hour, fractional offsets, and microsecond truncation remain intact.
        for instant in (
            "2024-03-10T06:59:59.999999+00:00",
            "2024-03-10T07:00:00.000001+00:00",
            "2024-11-03T05:30:00.123456+00:00",
            "2024-11-03T06:30:00.123456+00:00",
            "2024-04-06T15:00:00.999999+00:00",
            "2024-10-05T15:30:00.000001+00:00",
        ):
            stamp = datetime.fromisoformat(instant).timestamp()

            class FrozenClock:
                @staticmethod
                def now(tz=None, *, _stamp=stamp):
                    return datetime.fromtimestamp(_stamp, tz)

            for module in (clock, frontmatter, report):
                monkeypatch.setattr(module, "datetime", FrozenClock)
            old = datetime.fromtimestamp(stamp).astimezone()
            assert clock.now_iso() == old.isoformat(timespec="milliseconds")
            assert frontmatter.frontmatter_timestamp() == clock.now_iso()
            assert (
                report.build_report_shell(log_file=None)["generated_at"]
                == old.isoformat()
            )
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()
