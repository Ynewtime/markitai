"""Concurrency slots that pipeline conversion and LLM enhancement.

A batch that runs each file end to end in one slot alternates: while a
file waits on the model it holds a slot no conversion can use, so the model
idles between rounds while the next files convert. ``StagedSlots`` gives a
file a conversion slot and, once it is converted, trades it for an
LLM-stage slot (``convert_document_core`` awaits the trade through
``LLM_STAGE_HANDOFF``): the next file converts while this one waits on the
model. The LLM-stage slots also bound how many converted documents can wait
in memory for their turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from markitai.workflow.core import LLM_STAGE_HANDOFF


class StagedSlots:
    """Conversion slots handed on to LLM-stage slots at the LLM step."""

    def __init__(self, conversion: int, llm_stage: int) -> None:
        self._conversion = asyncio.Semaphore(max(1, conversion))
        self._llm_stage = asyncio.Semaphore(max(1, llm_stage))

    @asynccontextmanager
    async def slot(self, *, llm: bool) -> AsyncIterator[None]:
        """Hold a conversion slot for one document.

        Args:
            llm: The document gets LLM enhancement: its slot is handed on at
                the LLM step. Without it the slot is simply held throughout.
        """
        held = {"conversion": False, "llm": False}
        await self._conversion.acquire()
        held["conversion"] = True

        async def handoff() -> None:
            await self._llm_stage.acquire()
            held["llm"] = True
            self._conversion.release()
            held["conversion"] = False

        token = LLM_STAGE_HANDOFF.set(handoff if llm else None)
        try:
            yield
        finally:
            LLM_STAGE_HANDOFF.reset(token)
            if held["conversion"]:
                self._conversion.release()
            if held["llm"]:
                self._llm_stage.release()
