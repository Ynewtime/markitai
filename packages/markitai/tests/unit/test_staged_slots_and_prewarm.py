"""Pipelining helpers: staged batch slots and the LiteLLM prewarm."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from markitai.workflow.core import LLM_STAGE_HANDOFF
from markitai.workflow.slots import StagedSlots


class TestStagedSlots:
    @pytest.mark.asyncio
    async def test_the_next_file_converts_while_one_waits_on_the_model(
        self,
    ) -> None:
        slots = StagedSlots(conversion=1, llm_stage=2)
        order: list[str] = []
        first_in_llm = asyncio.Event()
        release_first = asyncio.Event()

        async def first() -> None:
            async with slots.slot(llm=True):
                order.append("first converts")
                handoff = LLM_STAGE_HANDOFF.get()
                assert handoff is not None
                await handoff()
                first_in_llm.set()
                await release_first.wait()
                order.append("first enhanced")

        async def second() -> None:
            await first_in_llm.wait()
            async with slots.slot(llm=True):
                order.append("second converts")
            release_first.set()

        await asyncio.wait_for(asyncio.gather(first(), second()), 5)

        # With one conversion slot, the second file got it while the first
        # was still in its LLM step
        assert order == ["first converts", "second converts", "first enhanced"]

    @pytest.mark.asyncio
    async def test_llm_stage_slots_bound_the_waiting_documents(self) -> None:
        slots = StagedSlots(conversion=3, llm_stage=1)
        in_llm = 0
        peak = 0

        async def document() -> None:
            nonlocal in_llm, peak
            async with slots.slot(llm=True):
                handoff = LLM_STAGE_HANDOFF.get()
                assert handoff is not None
                await handoff()
                in_llm += 1
                peak = max(peak, in_llm)
                await asyncio.sleep(0.01)
                in_llm -= 1

        await asyncio.wait_for(asyncio.gather(*(document() for _ in range(5))), 5)
        assert peak == 1

    @pytest.mark.asyncio
    async def test_without_llm_there_is_no_handoff_and_slots_come_back(
        self,
    ) -> None:
        slots = StagedSlots(conversion=1, llm_stage=1)
        for _ in range(3):  # a leaked slot would hang the second round
            async with slots.slot(llm=False):
                assert LLM_STAGE_HANDOFF.get() is None
        assert LLM_STAGE_HANDOFF.get() is None

    @pytest.mark.asyncio
    async def test_a_failure_after_the_handoff_releases_the_llm_slot(self) -> None:
        slots = StagedSlots(conversion=1, llm_stage=1)
        with pytest.raises(RuntimeError):
            async with slots.slot(llm=True):
                handoff = LLM_STAGE_HANDOFF.get()
                assert handoff is not None
                await handoff()
                raise RuntimeError("model call failed")
        async with slots.slot(llm=True):  # both kinds of slot are free again
            handoff = LLM_STAGE_HANDOFF.get()
            assert handoff is not None
            await asyncio.wait_for(handoff(), 1)


class TestPrewarm:
    @pytest.fixture(autouse=True)
    def fresh_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from markitai.utils import prewarm

        monkeypatch.setattr(prewarm, "_thread", None)
        monkeypatch.setattr(prewarm, "_done", False)
        monkeypatch.setattr(prewarm, "_callbacks", [])

    def test_callbacks_run_after_the_import(self) -> None:
        from markitai.utils.prewarm import prewarm_litellm

        ran = threading.Event()
        seen: list[bool] = []

        def after() -> None:
            import sys

            seen.append("litellm" in sys.modules)
            ran.set()

        prewarm_litellm(after=after)
        assert ran.wait(30)
        assert seen == [True]

    def test_a_callback_queued_after_the_import_still_runs(self) -> None:
        from markitai.utils import prewarm

        first = threading.Event()
        prewarm.prewarm_litellm(after=first.set)
        assert first.wait(30)
        assert prewarm._thread is not None
        prewarm._thread.join(5)

        second = threading.Event()
        prewarm.prewarm_litellm(after=second.set)
        assert second.wait(5)

    def test_a_failing_callback_does_not_stop_the_next(self) -> None:
        from markitai.utils.prewarm import prewarm_litellm

        done = threading.Event()

        def boom() -> None:
            raise RuntimeError("callback failed")

        prewarm_litellm(after=boom)
        prewarm_litellm(after=done.set)
        assert done.wait(30)


class TestDeferredVisionCheck:
    def _cfg(self, model: str) -> Any:
        from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.image.alt_enabled = True
        cfg.llm.model_list = [
            ModelConfig(model_name="default", litellm_params=LiteLLMParams(model=model))
        ]
        return cfg

    def test_the_lookup_waits_for_the_prewarm_and_warns_through_a_notice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.cli.processors import validators

        queued: list[Any] = []
        monkeypatch.setattr(
            "markitai.utils.prewarm.prewarm_litellm",
            lambda after=None: queued.append(after),
        )
        notices: list[str] = []
        monkeypatch.setattr(
            validators,
            "user_notice",
            lambda message, *args: notices.append(message.format(*args)),
        )

        validators.check_vision_model_config(
            self._cfg("openai/some-text-model"), MagicMock(), deferred=True
        )
        assert notices == []  # nothing looked up yet
        (callback,) = queued

        litellm = pytest.importorskip("litellm")
        monkeypatch.setattr(
            litellm, "get_model_info", lambda _model: {"supports_vision": False}
        )
        callback()
        (notice,) = notices
        assert "No vision-capable models detected" in notice
        assert "openai/some-text-model" in notice

    def test_a_vision_model_found_later_stays_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.cli.processors import validators

        litellm = pytest.importorskip("litellm")
        monkeypatch.setattr(
            litellm, "get_model_info", lambda _model: {"supports_vision": True}
        )
        notices: list[str] = []
        monkeypatch.setattr(
            validators, "user_notice", lambda message, *_args: notices.append(message)
        )

        validators._notice_if_no_vision_model(["openai/gpt-4o"], ["openai/gpt-4o"])

        assert notices == []
