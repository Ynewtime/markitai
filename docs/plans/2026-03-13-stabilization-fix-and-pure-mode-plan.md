# Stabilization Fix & Pure Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix duplicate stabilization warnings, add prompt tail reminders to reduce LLM drift, and introduce `--pure` mode for transparent LLM pipeline.

**Architecture:** Three independent changes sharing one commit history. Problem A removes redundant stabilization calls and adds a `paged_stabilized` flag. Problem B appends REMINDER text to vision prompts. Problem C adds a `pure` config field, CLI flag, and a new lightweight code path that bypasses all markitai processing.

**Tech Stack:** Python 3.13, Click (CLI), Pydantic v2 (config), pytest + AsyncMock (tests)

**Spec:** `docs/plans/2026-03-13-stabilization-fix-and-pure-mode-design.md`

---

## Chunk 1: Problem A — Deduplicate Stabilization Warnings

### Task 1: Remove redundant `maybe_stabilize_markdown` in `enhance_with_vision`

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/single.py:399-402`
- Test: `packages/markitai/tests/unit/test_workflow_single.py`

- [ ] **Step 1: Write the failing test**

The test verifies that `enhance_with_vision` does NOT call `maybe_stabilize_markdown` — the inner `enhance_document_complete` already handles stabilization.

```python
# In test_workflow_single.py — add this test class

class TestEnhanceWithVisionNoDoubleStabilize:
    """Verify enhance_with_vision does not call maybe_stabilize_markdown."""

    async def test_no_redundant_stabilize_call(self, tmp_path: Path):
        """enhance_with_vision should not call maybe_stabilize_markdown."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.config import MarkitaiConfig
        from markitai.workflow.single import SingleFileWorkflow

        config = MarkitaiConfig()
        mock_processor = MagicMock()
        mock_processor.enhance_document_complete = AsyncMock(
            return_value=("cleaned md", "---\ntitle: test\n---")
        )
        mock_processor.get_context_cost = MagicMock(return_value=0.001)
        mock_processor.get_context_usage = MagicMock(return_value={})

        workflow = SingleFileWorkflow(config, processor=mock_processor)

        page_images = [{"path": str(tmp_path / "page1.png"), "page": 1}]
        (tmp_path / "page1.png").write_bytes(b"fake")

        with patch(
            "markitai.workflow.single.maybe_stabilize_markdown"
        ) as mock_stabilize:
            result = await workflow.enhance_with_vision(
                "original md", page_images, source="test.pptx"
            )
            mock_stabilize.assert_not_called()

        assert result[0] == "cleaned md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py::TestEnhanceWithVisionNoDoubleStabilize -v`
Expected: FAIL — `maybe_stabilize_markdown` is currently called at single.py:401

- [ ] **Step 3: Remove the redundant call**

In `packages/markitai/src/markitai/workflow/single.py`, remove lines 399-402:

```python
# REMOVE these lines from enhance_with_vision:
            from markitai.workflow.helpers import maybe_stabilize_markdown

            cleaned_content = maybe_stabilize_markdown(
                self.processor, extracted_text, cleaned_content, source
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py::TestEnhanceWithVisionNoDoubleStabilize -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/single.py packages/markitai/tests/unit/test_workflow_single.py
git commit -m "fix: remove redundant maybe_stabilize_markdown in enhance_with_vision"
```

### Task 2: Remove redundant `maybe_stabilize_markdown` in `process_document_with_llm`

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/single.py:150-155`
- Test: `packages/markitai/tests/unit/test_workflow_single.py`

- [ ] **Step 1: Write the failing test**

```python
class TestProcessDocumentNoDoubleStabilize:
    """Verify process_document_with_llm does not call maybe_stabilize_markdown."""

    async def test_no_redundant_stabilize_call(self, tmp_path: Path):
        """process_document_with_llm should not call maybe_stabilize_markdown."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.config import MarkitaiConfig
        from markitai.workflow.single import SingleFileWorkflow

        config = MarkitaiConfig()
        mock_processor = MagicMock()
        mock_processor.process_document = AsyncMock(
            return_value=("cleaned md", "---\ntitle: test\n---")
        )
        mock_processor.format_llm_output = MagicMock(
            return_value="---\ntitle: test\n---\n\ncleaned md"
        )
        mock_processor.get_context_cost = MagicMock(return_value=0.001)
        mock_processor.get_context_usage = MagicMock(return_value={})

        workflow = SingleFileWorkflow(config, processor=mock_processor)

        output_file = tmp_path / "test.md"
        output_file.write_text("# original", encoding="utf-8")

        with patch(
            "markitai.workflow.single.maybe_stabilize_markdown"
        ) as mock_stabilize:
            await workflow.process_document_with_llm(
                "original md", "test.txt", output_file
            )
            mock_stabilize.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py::TestProcessDocumentNoDoubleStabilize -v`
Expected: FAIL

- [ ] **Step 3: Remove the redundant call**

In `packages/markitai/src/markitai/workflow/single.py`, remove lines 150-155:

```python
# REMOVE these lines from process_document_with_llm:
            from markitai.workflow.helpers import maybe_stabilize_markdown

            baseline_markdown = _read_markdown_body(output_file, markdown)
            cleaned = maybe_stabilize_markdown(
                self.processor, baseline_markdown, cleaned, source
            )
```

And change the variable name from the `process_document` return:

```python
            cleaned, frontmatter = await self.processor.process_document(
                markdown,
                source,
                title=title,
            )
```

The `cleaned` variable is already used correctly below — it flows into `format_llm_output`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py::TestProcessDocumentNoDoubleStabilize -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/single.py packages/markitai/tests/unit/test_workflow_single.py
git commit -m "fix: remove redundant maybe_stabilize_markdown in process_document_with_llm"
```

### Task 3: Add `paged_stabilized` flag and skip `stabilize_written_llm_output`

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py:44-92` (ConversionContext)
- Modify: `packages/markitai/src/markitai/workflow/core.py:914-954` (convert_document_core Step 7)
- Test: `packages/markitai/tests/unit/test_workflow_core.py`

- [ ] **Step 1: Write the failing test**

```python
class TestPagedStabilizedFlag:
    """Test that paged_stabilized flag skips stabilize_written_llm_output."""

    def test_context_has_paged_stabilized_field(self):
        """ConversionContext should have paged_stabilized field, default False."""
        from markitai.workflow.core import ConversionContext

        ctx = ConversionContext(
            input_path=Path("test.pptx"),
            output_dir=Path("/tmp/out"),
            config=MarkitaiConfig(),
        )
        assert ctx.paged_stabilized is False

    def test_stabilize_skipped_when_flag_set(self, tmp_path: Path):
        """stabilize_written_llm_output should be skipped when paged_stabilized=True."""
        from markitai.workflow.core import ConversionContext, stabilize_written_llm_output

        output_file = tmp_path / "test.md"
        output_file.write_text("# baseline", encoding="utf-8")
        llm_file = tmp_path / "test.llm.md"
        llm_file.write_text("---\ntitle: t\n---\n\n# changed", encoding="utf-8")

        ctx = ConversionContext(
            input_path=Path("test.pptx"),
            output_dir=tmp_path,
            config=MarkitaiConfig(),
        )
        ctx.output_file = output_file
        ctx.conversion_result = ConvertResult(markdown="# baseline", images=[], metadata={})
        ctx.paged_stabilized = True

        result = stabilize_written_llm_output(ctx, MagicMock())
        assert result is False  # Should skip, return False (no rewrite)

        # Verify .llm.md was NOT rewritten
        assert llm_file.read_text(encoding="utf-8") == "---\ntitle: t\n---\n\n# changed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestPagedStabilizedFlag -v`
Expected: FAIL — `ConversionContext` has no `paged_stabilized` field

- [ ] **Step 3: Implement the flag and guard**

In `packages/markitai/src/markitai/workflow/core.py`:

1. Add field to `ConversionContext` (after `use_multiprocess_images`):
```python
    paged_stabilized: bool = False
```

2. Add early return in `stabilize_written_llm_output` (after line 511):
```python
    if ctx.paged_stabilized:
        return False
```

3. Set `ctx.paged_stabilized = True` before each `stabilize_written_llm_output` call, since by those points the internal stabilization in document.py has already run:

In `process_with_standard_llm` (core.py), before line 761:
```python
        ctx.paged_stabilized = True
        stabilize_written_llm_output(ctx, processor)
```

In `convert_document_core` (core.py), before line 940 (vision path):
```python
            ctx.paged_stabilized = True
            stabilize_written_llm_output(ctx, ctx.shared_processor)
```

Both `stabilize_written_llm_output` calls will now be skipped by the early return guard added in step 3.2.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestPagedStabilizedFlag -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/workflow/core.py packages/markitai/tests/unit/test_workflow_core.py
git commit -m "fix: add paged_stabilized flag to skip redundant stabilization"
```

---

## Chunk 2: Problem B — Prompt Tail Reminders

### Task 4: Add REMINDER to `document_vision_user.md`

**Files:**
- Modify: `packages/markitai/src/markitai/prompts/document_vision_user.md`
- Test: `packages/markitai/tests/unit/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# In test_prompts.py — add this test

class TestVisionPromptReminder:
    """Verify document_vision_user prompt contains tail REMINDER."""

    def test_vision_user_prompt_has_reminder(self):
        """document_vision_user.md should end with a REMINDER about placeholders."""
        from markitai.llm.processor import PromptManager

        pm = PromptManager()
        prompt = pm.get_prompt("document_vision_user", content="test content")
        assert "REMINDER:" in prompt
        assert "__MARKITAI_" in prompt
        # REMINDER should appear AFTER the content
        content_pos = prompt.index("test content")
        reminder_pos = prompt.index("REMINDER:")
        assert reminder_pos > content_pos
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_prompts.py::TestVisionPromptReminder -v`
Expected: FAIL — no REMINDER in current prompt

- [ ] **Step 3: Add REMINDER to the prompt**

Append to `packages/markitai/src/markitai/prompts/document_vision_user.md`:

```markdown
Please clean the following document:

---

{content}

---

REMINDER: All `__MARKITAI_*__` placeholders must appear in your output exactly as in the input. Do not remove, modify, or merge any placeholder. Do not wrap output in a code block.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_prompts.py::TestVisionPromptReminder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/prompts/document_vision_user.md packages/markitai/tests/unit/test_prompts.py
git commit -m "fix: add placeholder REMINDER to vision user prompt tail"
```

### Task 5: Add REMINDER to content_parts in `enhance_document_with_vision` and `_enhance_with_frontmatter`

**Files:**
- Modify: `packages/markitai/src/markitai/llm/document.py:1054-1077` (enhance_document_with_vision)
- Modify: `packages/markitai/src/markitai/llm/document.py:1348-1367` (_enhance_with_frontmatter)
- Test: `packages/markitai/tests/unit/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# In test_llm.py — add this test class

class TestVisionContentPartsReminder:
    """Verify content_parts ends with REMINDER text part."""

    async def test_enhance_with_vision_appends_reminder(self):
        """enhance_document_with_vision should append REMINDER to content_parts."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.config import LLMConfig, LiteLLMParams, ModelConfig, PromptsConfig
        from markitai.llm import LLMProcessor

        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test"
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, PromptsConfig())

        # Capture the messages passed to _call_llm
        captured_messages = []

        async def capture_call_llm(model, messages, context=""):
            captured_messages.append(messages)
            return MagicMock(content="cleaned output")

        processor._call_llm = capture_call_llm
        processor._persistent_cache = MagicMock(get=MagicMock(return_value=None), set=MagicMock())
        processor._get_cached_image = MagicMock(return_value=("image/png", "base64data"))
        processor._stabilize_paged_markdown = MagicMock(side_effect=lambda orig, cleaned, ctx: cleaned)

        from pathlib import Path

        result = await processor.enhance_document_with_vision(
            "<!-- Slide number: 1 -->\n# Slide 1",
            [Path("/tmp/fake.png")],
            context="test.pptx",
        )

        assert len(captured_messages) == 1
        user_content = captured_messages[0][1]["content"]
        # Last text part should be the REMINDER
        text_parts = [p for p in user_content if p.get("type") == "text"]
        last_text = text_parts[-1]["text"]
        assert "REMINDER" in last_text
        assert "__MARKITAI_" in last_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_llm.py::TestVisionContentPartsReminder -v`
Expected: FAIL — no REMINDER in content_parts

- [ ] **Step 3: Add REMINDER to both methods**

In `packages/markitai/src/markitai/llm/document.py`:

**In `enhance_document_with_vision`**, after the for loop that adds page images (after line 1076), add:

```python
        # Append tail reminder to reinforce placeholder rules for last pages
        content_parts.append(
            {
                "type": "text",
                "text": "\nREMINDER: Preserve ALL __MARKITAI_*__ placeholders exactly as-is. "
                "Do not remove or modify any placeholder. "
                "Output every page/slide — do not skip the last pages.",
            }
        )
```

**In `_enhance_with_frontmatter`**, after the for loop that adds page images (after line 1366), add the same block:

```python
        # Append tail reminder to reinforce placeholder rules for last pages
        content_parts.append(
            {
                "type": "text",
                "text": "\nREMINDER: Preserve ALL __MARKITAI_*__ placeholders exactly as-is. "
                "Do not remove or modify any placeholder. "
                "Output every page/slide — do not skip the last pages.",
            }
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_llm.py::TestVisionContentPartsReminder -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/llm/document.py packages/markitai/tests/unit/test_llm.py
git commit -m "fix: add REMINDER to vision content_parts tail to reduce LLM drift"
```

---

## Chunk 3: Problem C — Pure Mode

### Task 6: Add `pure` field to `LLMConfig`

**Files:**
- Modify: `packages/markitai/src/markitai/config.py:241-248`
- Test: `packages/markitai/tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# In test_config.py — add this test class

class TestPureModeConfig:
    """Test pure mode configuration."""

    def test_llm_config_has_pure_field_default_false(self):
        """LLMConfig.pure should default to False."""
        from markitai.config import LLMConfig

        config = LLMConfig()
        assert config.pure is False

    def test_llm_config_pure_can_be_set(self):
        """LLMConfig.pure can be set to True."""
        from markitai.config import LLMConfig

        config = LLMConfig(pure=True)
        assert config.pure is True

    def test_markitai_config_json_roundtrip(self, tmp_path: Path):
        """Pure mode should survive JSON serialization."""
        from markitai.config import MarkitaiConfig

        config = MarkitaiConfig()
        config.llm.pure = True
        data = config.model_dump()
        assert data["llm"]["pure"] is True

        restored = MarkitaiConfig(**data)
        assert restored.llm.pure is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_config.py::TestPureModeConfig -v`
Expected: FAIL — `LLMConfig` has no `pure` field

- [ ] **Step 3: Add the field**

In `packages/markitai/src/markitai/config.py`, add to `LLMConfig`:

```python
class LLMConfig(BaseModel):
    """LLM configuration."""

    enabled: bool = False
    pure: bool = Field(
        default=False,
        description="Pure mode: raw MD sent directly to LLM, no markitai processing",
    )
    model_list: list[ModelConfig] = Field(default_factory=list)
    router_settings: RouterSettings = Field(default_factory=RouterSettings)
    concurrency: int = DEFAULT_LLM_CONCURRENCY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_config.py::TestPureModeConfig -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/config.py packages/markitai/tests/unit/test_config.py
git commit -m "feat: add pure field to LLMConfig"
```

### Task 7: Add `--pure` CLI flag

**Files:**
- Modify: `packages/markitai/src/markitai/cli/main.py:99-294` (option decoration)
- Modify: `packages/markitai/src/markitai/cli/main.py:460-492` (option application)
- Test: `packages/markitai/tests/unit/test_cli_main.py` or `packages/markitai/tests/unit/cli/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/unit/cli/test_main.py — add this test class

class TestPureCLIFlag:
    """Test --pure CLI flag."""

    def test_pure_flag_sets_config(self, cli_runner, tmp_path: Path):
        """--pure should set config.llm.pure=True and imply --llm."""
        from click.testing import CliRunner

        from markitai.cli.main import app

        runner = CliRunner()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")

        # Use --dry-run to avoid actual processing
        result = runner.invoke(
            app,
            [str(txt_file), "--pure", "--dry-run", "-o", str(tmp_path / "out")],
            catch_exceptions=False,
        )
        # --pure implies --llm, so LLM should be enabled
        # Dry run just validates — we check it doesn't error
        # The actual config validation happens in the app function
        assert result.exit_code == 0 or "no models configured" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_main.py::TestPureCLIFlag -v`
Expected: FAIL — `--pure` is not a recognized option

- [ ] **Step 3: Add the CLI flag**

In `packages/markitai/src/markitai/cli/main.py`:

1. Add the `@click.option` decorator (after `--dry-run`, before `--interactive`):

```python
@click.option(
    "--pure",
    is_flag=True,
    help="Pure mode: raw MD → LLM → output, no markitai processing (implies --llm).",
)
```

2. Add `pure: bool` to the `app` function signature (after `dry_run: bool`):

```python
def app(
    ctx: Context,
    ...
    dry_run: bool,
    pure: bool,  # ADD
) -> None:
```

3. In the option application section (after `screenshot_only` handling, around line 477), add:

```python
    if pure:
        cfg.llm.pure = True
        cfg.llm.enabled = True  # --pure implies --llm
```

4. Also support `MARKITAI_PURE` env var. Add in the same section:

```python
    # Env var support for pure mode
    if not pure and os.environ.get("MARKITAI_PURE", "").strip() in ("1", "true", "yes"):
        cfg.llm.pure = True
        cfg.llm.enabled = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_main.py::TestPureCLIFlag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/main.py packages/markitai/tests/unit/cli/test_main.py
git commit -m "feat: add --pure CLI flag with MARKITAI_PURE env support"
```

### Task 8: Add `clean_document_pure` to DocumentMixin

**Files:**
- Modify: `packages/markitai/src/markitai/llm/document.py`
- Test: `packages/markitai/tests/unit/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# In test_llm.py — add this test class

class TestCleanDocumentPure:
    """Test clean_document_pure method."""

    async def test_pure_sends_raw_markdown_to_llm(self):
        """clean_document_pure should send raw markdown without any processing."""
        from unittest.mock import AsyncMock, MagicMock

        from markitai.config import LLMConfig, LiteLLMParams, ModelConfig, PromptsConfig
        from markitai.llm import LLMProcessor

        config = LLMConfig(
            enabled=True,
            pure=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test"
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, PromptsConfig())

        captured_messages = []

        async def capture_call_llm(model, messages, context=""):
            captured_messages.append(messages)
            return MagicMock(content="LLM cleaned output")

        processor._call_llm = capture_call_llm

        raw_md = "# Title\n\nSome **raw** content with ![img](path.jpg)"
        result = await processor.clean_document_pure(raw_md, "test.md")

        assert result == "LLM cleaned output"
        assert len(captured_messages) == 1
        messages = captured_messages[0]
        # System prompt should be cleaner_system
        assert "Markdown" in messages[0]["content"]
        # User prompt should contain the raw markdown as-is
        assert raw_md in messages[1]["content"]

    async def test_pure_no_protection_no_stabilization(self):
        """clean_document_pure should NOT call protect_content or stabilize."""
        from unittest.mock import AsyncMock, MagicMock, call, patch

        from markitai.config import LLMConfig, LiteLLMParams, ModelConfig, PromptsConfig
        from markitai.llm import LLMProcessor

        config = LLMConfig(
            enabled=True,
            pure=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test"
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, PromptsConfig())

        async def fake_call_llm(model, messages, context=""):
            return MagicMock(content="output")

        processor._call_llm = fake_call_llm

        with patch.object(processor, "protect_content") as mock_protect, \
             patch.object(processor, "_stabilize_paged_markdown") as mock_stabilize:
            await processor.clean_document_pure("# Test", "test.md")
            mock_protect.assert_not_called()
            mock_stabilize.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_llm.py::TestCleanDocumentPure -v`
Expected: FAIL — `LLMProcessor` has no `clean_document_pure` method

- [ ] **Step 3: Implement `clean_document_pure`**

In `packages/markitai/src/markitai/llm/document.py`, add method to `DocumentMixin` (after `process_document`, around line 1750):

```python
    async def clean_document_pure(self, markdown: str, source: str) -> str:
        """Pure cleaning: send raw markdown to LLM, return response as-is.

        No content protection, no stabilization, no truncation, no frontmatter.
        The LLM decides what to clean based on the cleaner prompt.

        Args:
            markdown: Raw markdown content
            source: Source file name for logging context

        Returns:
            LLM response content as-is
        """
        system_prompt = self._prompt_manager.get_prompt("cleaner_system")
        user_prompt = self._prompt_manager.get_prompt("cleaner_user", content=markdown)
        response = await self._call_llm(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            context=source,
        )
        return response.content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_llm.py::TestCleanDocumentPure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/llm/document.py packages/markitai/tests/unit/test_llm.py
git commit -m "feat: add clean_document_pure to DocumentMixin"
```

### Task 9: Add `process_document_pure` to SingleFileWorkflow

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/single.py`
- Test: `packages/markitai/tests/unit/test_workflow_single.py`

- [ ] **Step 1: Write the failing test**

```python
# In test_workflow_single.py — add this test class

class TestProcessDocumentPure:
    """Test process_document_pure method."""

    async def test_pure_writes_llm_output_directly(self, tmp_path: Path):
        """process_document_pure should write LLM response directly to .llm.md."""
        from unittest.mock import AsyncMock, MagicMock

        from markitai.config import MarkitaiConfig
        from markitai.workflow.single import SingleFileWorkflow

        config = MarkitaiConfig()
        mock_processor = MagicMock()
        mock_processor.clean_document_pure = AsyncMock(return_value="LLM cleaned output")
        mock_processor.get_context_cost = MagicMock(return_value=0.001)
        mock_processor.get_context_usage = MagicMock(return_value={"model": {"requests": 1}})

        workflow = SingleFileWorkflow(config, processor=mock_processor)

        output_file = tmp_path / "test.md"
        output_file.write_text("# original", encoding="utf-8")

        markdown, cost, usage = await workflow.process_document_pure(
            "# original markdown", "test.txt", output_file
        )

        # Verify .llm.md written with raw LLM output (no frontmatter wrapping)
        llm_file = tmp_path / "test.llm.md"
        assert llm_file.exists()
        content = llm_file.read_text(encoding="utf-8")
        assert content == "LLM cleaned output"

        # Verify return values
        assert markdown == "# original markdown"  # Original returned unchanged
        assert cost == 0.001
        assert usage == {"model": {"requests": 1}}

    async def test_pure_does_not_call_format_llm_output(self, tmp_path: Path):
        """process_document_pure should NOT call format_llm_output."""
        from unittest.mock import AsyncMock, MagicMock

        from markitai.config import MarkitaiConfig
        from markitai.workflow.single import SingleFileWorkflow

        config = MarkitaiConfig()
        mock_processor = MagicMock()
        mock_processor.clean_document_pure = AsyncMock(return_value="output")
        mock_processor.get_context_cost = MagicMock(return_value=0.0)
        mock_processor.get_context_usage = MagicMock(return_value={})

        workflow = SingleFileWorkflow(config, processor=mock_processor)

        output_file = tmp_path / "test.md"
        output_file.write_text("# x", encoding="utf-8")

        await workflow.process_document_pure("# x", "test.txt", output_file)
        mock_processor.format_llm_output.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py::TestProcessDocumentPure -v`
Expected: FAIL — `SingleFileWorkflow` has no `process_document_pure` method

- [ ] **Step 3: Implement `process_document_pure`**

In `packages/markitai/src/markitai/workflow/single.py`, add method to `SingleFileWorkflow` (after `process_document_with_llm`):

```python
    async def process_document_pure(
        self,
        markdown: str,
        source: str,
        output_file: Path,
    ) -> tuple[str, float, dict[str, dict[str, Any]]]:
        """Pure mode: send raw markdown to LLM, write response as-is.

        No ContentProtection, stabilization, frontmatter, or post-processing.

        Args:
            markdown: Raw markdown content
            source: Source file name
            output_file: Output file path (.md — .llm.md is derived from this)

        Returns:
            Tuple of (original_markdown, cost_usd, llm_usage)
        """
        try:
            cleaned = await self.processor.clean_document_pure(markdown, source)
            llm_output = output_file.with_suffix(".llm.md")
            atomic_write_text(llm_output, cleaned)
            logger.info(f"Written LLM version (pure): {llm_output}")
            cost = self.processor.get_context_cost(source)
            usage = self.processor.get_context_usage(source)
            return markdown, cost, usage
        except Exception as e:
            logger.error(f"Pure LLM processing failed: {format_error_message(e)}")
            return markdown, 0.0, {}
```

Ensure `atomic_write_text` is imported (already imported in single.py for existing methods).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py::TestProcessDocumentPure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/single.py packages/markitai/tests/unit/test_workflow_single.py
git commit -m "feat: add process_document_pure to SingleFileWorkflow"
```

### Task 10: Add `process_with_pure_llm` and wire into pipeline

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py`
- Test: `packages/markitai/tests/unit/test_workflow_core.py`

- [ ] **Step 1: Write the failing test**

```python
# In test_workflow_core.py — add this test class

class TestProcessWithPureLLM:
    """Test pure mode pipeline integration."""

    async def test_pure_mode_calls_process_document_pure(self, tmp_path: Path):
        """convert_document_core with pure=True should call process_document_pure."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.config import LLMConfig, LiteLLMParams, MarkitaiConfig, ModelConfig
        from markitai.converter.base import ConvertResult
        from markitai.workflow.core import ConversionContext, process_with_pure_llm

        config = MarkitaiConfig()
        config.llm.enabled = True
        config.llm.pure = True
        config.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o-mini", api_key="t"),
            )
        ]

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")
        output_file = tmp_path / "test.md"
        output_file.write_text("# Hello", encoding="utf-8")

        ctx = ConversionContext(
            input_path=txt_file,
            output_dir=tmp_path,
            config=config,
        )
        ctx.output_file = output_file
        ctx.conversion_result = ConvertResult(markdown="# Hello", images=[], metadata={})

        mock_workflow = MagicMock()
        mock_workflow.process_document_pure = AsyncMock(
            return_value=("# Hello", 0.001, {})
        )

        with patch("markitai.workflow.core.SingleFileWorkflow", return_value=mock_workflow), \
             patch("markitai.workflow.core.create_llm_processor"):
            result = await process_with_pure_llm(ctx)

        assert result.success
        mock_workflow.process_document_pure.assert_called_once()
        assert ctx.llm_cost == 0.001

    async def test_pure_mode_skips_vision_and_alt(self, tmp_path: Path):
        """Pure mode should not call process_with_vision_llm or analyze_images."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.config import LLMConfig, LiteLLMParams, MarkitaiConfig, ModelConfig
        from markitai.converter.base import ConvertResult
        from markitai.workflow.core import ConversionContext, convert_document_core

        config = MarkitaiConfig()
        config.llm.enabled = True
        config.llm.pure = True
        config.image.alt_enabled = True  # Should be ignored in pure mode
        config.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o-mini", api_key="t"),
            )
        ]

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")

        ctx = ConversionContext(
            input_path=txt_file,
            output_dir=tmp_path,
            config=config,
        )

        with patch("markitai.workflow.core.process_with_pure_llm", new_callable=AsyncMock, return_value=MagicMock(success=True)) as mock_pure, \
             patch("markitai.workflow.core.process_with_vision_llm") as mock_vision, \
             patch("markitai.workflow.core.process_with_standard_llm") as mock_standard, \
             patch("markitai.workflow.core.validate_and_detect_format", return_value=MagicMock(success=True)), \
             patch("markitai.workflow.core.prepare_output_directory", return_value=MagicMock(success=True)), \
             patch("markitai.workflow.core.resolve_output_file", return_value=MagicMock(success=True, skip_reason=None)), \
             patch("markitai.workflow.core.convert_document", new_callable=AsyncMock, return_value=MagicMock(success=True)), \
             patch("markitai.workflow.core.process_embedded_images", new_callable=AsyncMock, return_value=MagicMock(success=True)), \
             patch("markitai.workflow.core.write_base_markdown", return_value=MagicMock(success=True)):
            # Need to set conversion_result after convert_document runs
            ctx.conversion_result = ConvertResult(
                markdown="# Hello",
                images=[],
                metadata={"page_images": [{"path": "/tmp/p1.png", "page": 1}]},
            )
            result = await convert_document_core(ctx, 500_000_000)

        mock_pure.assert_called_once()
        mock_vision.assert_not_called()
        mock_standard.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestProcessWithPureLLM -v`
Expected: FAIL — `process_with_pure_llm` does not exist

- [ ] **Step 3: Implement `process_with_pure_llm` and wire into pipeline**

In `packages/markitai/src/markitai/workflow/core.py`:

1. Add `process_with_pure_llm` function (before `process_with_vision_llm`):

```python
async def process_with_pure_llm(ctx: ConversionContext) -> ConversionStepResult:
    """Pure mode: send raw markdown to LLM, write response as-is.

    No ContentProtection, stabilization, frontmatter, vision, or image analysis.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(ctx.config, processor=processor)

    try:
        ctx.conversion_result.markdown, doc_cost, doc_usage = (
            await workflow.process_document_pure(
                ctx.conversion_result.markdown,
                ctx.input_path.name,
                ctx.output_file,
            )
        )
        ctx.llm_cost += doc_cost
        merge_llm_usage(ctx.llm_usage, doc_usage)
    except Exception as e:
        return ConversionStepResult(
            success=False,
            error=f"Pure LLM processing failed: {format_error_message(e)}",
        )

    return ConversionStepResult(success=True)
```

2. In `convert_document_core`, modify Step 7 (around line 900):

```python
    # Step 7: LLM processing (if enabled)
    if ctx.config.llm.enabled and ctx.conversion_result is not None:
        # Ensure shared processor exists
        if ctx.shared_processor is None:
            from markitai.workflow.helpers import create_llm_processor
            ctx.shared_processor = create_llm_processor(ctx.config)

        if ctx.config.llm.pure and not ctx.config.screenshot.screenshot_only:
            # Pure mode: raw MD → LLM → .llm.md, nothing else
            # --screenshot-only takes precedence over --pure (mutually exclusive)
            result = await process_with_pure_llm(ctx)
            if not result.success:
                return result
        else:
            # Existing logic unchanged...
            page_images = ctx.conversion_result.metadata.get("page_images", [])
            has_page_images = len(page_images) > 0
            # ... rest of existing code ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestProcessWithPureLLM -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest packages/markitai/tests/unit/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Run linting and type checking**

Run: `uv run ruff check --fix && uv run ruff format`
Run: `uv run pyright packages/markitai/src/markitai/workflow/core.py packages/markitai/src/markitai/workflow/single.py packages/markitai/src/markitai/llm/document.py packages/markitai/src/markitai/config.py`

- [ ] **Step 7: Commit**

```bash
git add packages/markitai/src/markitai/workflow/core.py packages/markitai/tests/unit/test_workflow_core.py
git commit -m "feat: add process_with_pure_llm and wire pure mode into pipeline"
```

---

## Post-Implementation Verification

- [ ] **Run full test suite with coverage**

```bash
uv run pytest packages/markitai/tests/unit/ --cov=markitai -q
```

- [ ] **Run pre-commit hooks**

```bash
uv run pre-commit run --all-files
```

- [ ] **Manual smoke test (optional, requires LLM API)**

```bash
# Pure mode
markitai test.txt --pure -o /tmp/out

# Verify .llm.md has no frontmatter, is raw LLM output
cat /tmp/out/test.llm.md

# Normal mode (should still work)
markitai test.txt --preset rich -o /tmp/out2
```
