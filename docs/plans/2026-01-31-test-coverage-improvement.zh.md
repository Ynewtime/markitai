# 测试覆盖率提升实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 将 Markitai 测试覆盖率从 53% 提升到 80%

**架构：** 采用分层测试策略——优先测试纯函数（高 ROI），然后扩展到 CLI 命令测试（使用 CliRunner），最后添加 Mock 外部依赖的测试。

**技术栈：** pytest, pytest-cov, pytest-asyncio, Click CliRunner, unittest.mock

---

## Task 1: 测试 `format_size()` 纯函数

**Files:**
- Test: `packages/markitai/tests/unit/test_cache_cli.py` (Create)
- Source: `packages/markitai/src/markitai/cli/commands/cache.py:26-33`

**Step 1: 创建测试文件并编写失败测试**

创建 `packages/markitai/tests/unit/test_cache_cli.py`:

```python
"""Unit tests for cache CLI commands."""

from __future__ import annotations

import pytest

from markitai.cli.commands.cache import format_size


class TestFormatSize:
    """Tests for format_size function."""

    @pytest.mark.parametrize(
        "bytes_val,expected",
        [
            (0, "0 B"),
            (1, "1 B"),
            (512, "512 B"),
            (1023, "1023 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024 * 1024, "1.00 MB"),
            (1024 * 1024 * 10, "10.00 MB"),
            (1024 * 1024 * 1024, "1024.00 MB"),
        ],
    )
    def test_format_size(self, bytes_val: int, expected: str) -> None:
        """Test format_size with various byte values."""
        assert format_size(bytes_val) == expected

    def test_format_size_boundary_kb(self) -> None:
        """Test KB boundary (1024 bytes)."""
        assert format_size(1023) == "1023 B"
        assert format_size(1024) == "1.0 KB"

    def test_format_size_boundary_mb(self) -> None:
        """Test MB boundary (1024 * 1024 bytes)."""
        assert format_size(1024 * 1024 - 1) == "1024.0 KB"
        assert format_size(1024 * 1024) == "1.00 MB"
```

**Step 2: 运行测试验证通过**

Run: `cd packages/markitai && uv run pytest tests/unit/test_cache_cli.py -v`
Expected: PASS (format_size 已在源文件中实现)

**Step 3: 提交**

```bash
git add packages/markitai/tests/unit/test_cache_cli.py
git commit -m "test: add unit tests for cache CLI format_size function"
```

---

## Task 2: 测试 `_normalize_bypass_list()` 函数

**Files:**
- Modify: `packages/markitai/tests/unit/test_fetch.py`
- Source: `packages/markitai/src/markitai/fetch.py:706-781`

**Step 1: 添加测试类**

在 `packages/markitai/tests/unit/test_fetch.py` 末尾添加：

```python
class TestNormalizeBypassList:
    """Tests for _normalize_bypass_list function."""

    def test_empty_input(self) -> None:
        """Test with empty input."""
        from markitai.fetch import _normalize_bypass_list

        assert _normalize_bypass_list("") == ""

    def test_windows_local_marker_removed(self) -> None:
        """Test that <local> marker is removed."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("<local>,localhost")
        assert "<local>" not in result
        assert "localhost" in result

    def test_wildcard_domain_normalized(self) -> None:
        """Test *.domain.com -> .domain.com conversion."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("*.example.com")
        assert result == ".example.com"

    def test_wildcard_prefix_normalized(self) -> None:
        """Test *-prefix.domain.com extraction."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("*-internal.company.com")
        assert result == ".company.com"

    def test_ip_wildcard_to_cidr(self) -> None:
        """Test IP wildcard to CIDR conversion."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("127.*")
        assert result == "127.0.0.0/8"

        result = _normalize_bypass_list("10.*")
        assert result == "10.0.0.0/8"

        result = _normalize_bypass_list("192.168.*")
        assert result == "192.168.0.0/16"

    def test_172_range_to_cidr(self) -> None:
        """Test 172.16-31.* to CIDR conversion."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("172.16.*")
        assert result == "172.16.0.0/12"

        result = _normalize_bypass_list("172.31.*")
        assert result == "172.16.0.0/12"

    def test_partial_ip_wildcard(self) -> None:
        """Test partial IP wildcards like 100.64.*."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("100.64.*")
        assert result == "100.64"

    def test_multiple_entries_deduplicated(self) -> None:
        """Test that duplicate entries are removed."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("*.example.com,*.example.com,localhost")
        entries = result.split(",")
        assert len(entries) == 2
        assert ".example.com" in entries
        assert "localhost" in entries

    def test_plain_hostname_preserved(self) -> None:
        """Test that plain hostnames are preserved."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("localhost,myserver.local")
        assert "localhost" in result
        assert "myserver.local" in result

    def test_complex_bypass_list(self) -> None:
        """Test complex bypass list with multiple types."""
        from markitai.fetch import _normalize_bypass_list

        input_list = "<local>,*.internal.corp,127.*,192.168.*,myhost"
        result = _normalize_bypass_list(input_list)

        assert "<local>" not in result
        assert ".internal.corp" in result
        assert "127.0.0.0/8" in result
        assert "192.168.0.0/16" in result
        assert "myhost" in result
```

**Step 2: 运行测试验证通过**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py::TestNormalizeBypassList -v`
Expected: PASS

**Step 3: 提交**

```bash
git add packages/markitai/tests/unit/test_fetch.py
git commit -m "test: add tests for _normalize_bypass_list proxy function"
```

---

## Task 3: 测试 `DocumentMixin._protect_image_positions()` 和 `_restore_image_positions()`

**Files:**
- Create: `packages/markitai/tests/unit/test_document_utils.py`
- Source: `packages/markitai/src/markitai/llm/document.py:197-239`

**Step 1: 创建测试文件**

创建 `packages/markitai/tests/unit/test_document_utils.py`:

```python
"""Unit tests for document processing utilities."""

from __future__ import annotations

import pytest

from markitai.llm.document import DocumentMixin


class TestProtectImagePositions:
    """Tests for DocumentMixin._protect_image_positions static method."""

    def test_single_image(self) -> None:
        """Test protecting a single image reference."""
        text = "Before ![alt text](image.jpg) after"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert "![alt text](image.jpg)" not in protected
        assert "__MARKITAI_IMG_0__" in protected
        assert len(mapping) == 1
        assert mapping["__MARKITAI_IMG_0__"] == "![alt text](image.jpg)"

    def test_multiple_images(self) -> None:
        """Test protecting multiple image references."""
        text = "![img1](a.jpg) text ![img2](b.png) more ![img3](c.gif)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert len(mapping) == 3
        assert "__MARKITAI_IMG_0__" in protected
        assert "__MARKITAI_IMG_1__" in protected
        assert "__MARKITAI_IMG_2__" in protected

    def test_no_images(self) -> None:
        """Test with no images in text."""
        text = "Just plain text without images"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert protected == text
        assert len(mapping) == 0

    def test_screenshots_excluded(self) -> None:
        """Test that screenshots/ paths are excluded from protection."""
        text = "![screenshot](screenshots/page.jpg) and ![regular](image.png)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        # Screenshot should remain, regular image should be protected
        assert "![screenshot](screenshots/page.jpg)" in protected
        assert "![regular](image.png)" not in protected
        assert len(mapping) == 1

    def test_external_url_images(self) -> None:
        """Test protection of external URL images."""
        text = "![logo](https://example.com/logo.png)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert len(mapping) == 1
        assert "https://example.com/logo.png" in mapping["__MARKITAI_IMG_0__"]

    def test_empty_alt_text(self) -> None:
        """Test image with empty alt text."""
        text = "![](image.jpg)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert len(mapping) == 1
        assert mapping["__MARKITAI_IMG_0__"] == "![](image.jpg)"


class TestRestoreImagePositions:
    """Tests for DocumentMixin._restore_image_positions static method."""

    def test_restore_single_image(self) -> None:
        """Test restoring a single image."""
        mapping = {"__MARKITAI_IMG_0__": "![alt](image.jpg)"}
        text = "Before __MARKITAI_IMG_0__ after"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        assert restored == "Before ![alt](image.jpg) after"

    def test_restore_multiple_images(self) -> None:
        """Test restoring multiple images."""
        mapping = {
            "__MARKITAI_IMG_0__": "![img1](a.jpg)",
            "__MARKITAI_IMG_1__": "![img2](b.png)",
        }
        text = "Start __MARKITAI_IMG_0__ middle __MARKITAI_IMG_1__ end"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        assert "![img1](a.jpg)" in restored
        assert "![img2](b.png)" in restored
        assert "__MARKITAI_IMG_" not in restored

    def test_restore_empty_mapping(self) -> None:
        """Test restoration with empty mapping."""
        text = "No images here"
        restored = DocumentMixin._restore_image_positions(text, {})

        assert restored == text

    def test_roundtrip(self) -> None:
        """Test protect -> restore roundtrip."""
        original = "![first](1.jpg) text ![second](2.png) end"
        protected, mapping = DocumentMixin._protect_image_positions(original)
        restored = DocumentMixin._restore_image_positions(protected, mapping)

        assert restored == original
```

**Step 2: 运行测试验证通过**

Run: `cd packages/markitai && uv run pytest tests/unit/test_document_utils.py -v`
Expected: PASS

**Step 3: 提交**

```bash
git add packages/markitai/tests/unit/test_document_utils.py
git commit -m "test: add tests for DocumentMixin image position protection"
```

---

## Task 4: 测试 `CopilotProvider._messages_to_prompt()` 和 `_extract_json_from_response()`

**Files:**
- Modify: `packages/markitai/tests/unit/test_providers.py`
- Source: `packages/markitai/src/markitai/providers/copilot.py:305-520`

**Step 1: 添加测试类**

在 `packages/markitai/tests/unit/test_providers.py` 末尾添加：

```python
class TestCopilotProviderHelpers:
    """Tests for CopilotProvider helper methods."""

    @pytest.fixture
    def provider(self) -> Any:
        """Create a CopilotProvider instance for testing."""
        from markitai.providers.copilot import CopilotProvider, LITELLM_AVAILABLE

        if not LITELLM_AVAILABLE:
            pytest.skip("LiteLLM not available")
        return CopilotProvider()

    def test_messages_to_prompt_single_user(self, provider: Any) -> None:
        """Test converting single user message."""
        messages = [{"role": "user", "content": "Hello"}]
        prompt = provider._messages_to_prompt(messages)

        assert "Hello" in prompt

    def test_messages_to_prompt_system_message(self, provider: Any) -> None:
        """Test converting system message with tags."""
        messages = [{"role": "system", "content": "You are helpful."}]
        prompt = provider._messages_to_prompt(messages)

        assert "<system>" in prompt
        assert "You are helpful." in prompt
        assert "</system>" in prompt

    def test_messages_to_prompt_multi_turn(self, provider: Any) -> None:
        """Test converting multi-turn conversation."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = provider._messages_to_prompt(messages)

        assert "<system>" in prompt
        assert "Be concise." in prompt
        assert "Hi" in prompt
        assert "<assistant>" in prompt
        assert "Hello!" in prompt
        assert "How are you?" in prompt

    def test_messages_to_prompt_multimodal_extracts_text(self, provider: Any) -> None:
        """Test that multimodal messages extract only text."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            }
        ]
        prompt = provider._messages_to_prompt(messages)

        assert "Describe this:" in prompt
        assert "data:image" not in prompt

    def test_extract_json_from_markdown_code_block(self, provider: Any) -> None:
        """Test extracting JSON from markdown code block."""
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        json_str = provider._extract_json_from_response(text)

        assert json_str == '{"key": "value"}'

    def test_extract_json_from_bare_code_block(self, provider: Any) -> None:
        """Test extracting JSON from code block without json tag."""
        text = 'Result:\n```\n{"name": "test"}\n```'
        json_str = provider._extract_json_from_response(text)

        assert json_str == '{"name": "test"}'

    def test_extract_json_direct(self, provider: Any) -> None:
        """Test extracting JSON when it's the entire response."""
        text = '{"key": "value"}'
        json_str = provider._extract_json_from_response(text)

        assert json_str == '{"key": "value"}'

    def test_extract_json_array(self, provider: Any) -> None:
        """Test extracting JSON array."""
        text = '[1, 2, 3]'
        json_str = provider._extract_json_from_response(text)

        assert json_str == '[1, 2, 3]'

    def test_extract_json_embedded(self, provider: Any) -> None:
        """Test extracting embedded JSON object."""
        text = 'The answer is {"result": true} as expected.'
        json_str = provider._extract_json_from_response(text)

        assert json_str == '{"result": true}'

    def test_extract_json_no_json_returns_original(self, provider: Any) -> None:
        """Test that non-JSON text is returned as-is."""
        text = "This is just plain text without JSON."
        json_str = provider._extract_json_from_response(text)

        assert json_str == text

    def test_has_images_with_images(self, provider: Any) -> None:
        """Test _has_images with image content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            }
        ]
        assert provider._has_images(messages) is True

    def test_has_images_text_only(self, provider: Any) -> None:
        """Test _has_images with text-only content."""
        messages = [{"role": "user", "content": "Hello"}]
        assert provider._has_images(messages) is False

    def test_has_images_multipart_text_only(self, provider: Any) -> None:
        """Test _has_images with multipart text content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            }
        ]
        assert provider._has_images(messages) is False
```

**Step 2: 运行测试验证通过**

Run: `cd packages/markitai && uv run pytest tests/unit/test_providers.py::TestCopilotProviderHelpers -v`
Expected: PASS

**Step 3: 提交**

```bash
git add packages/markitai/tests/unit/test_providers.py
git commit -m "test: add tests for CopilotProvider helper methods"
```

---

## Task 5: 测试 `ClaudeAgentProvider._has_images()` 和相关方法

**Files:**
- Modify: `packages/markitai/tests/unit/test_providers.py`
- Source: `packages/markitai/src/markitai/providers/claude_agent.py`

**Step 1: 先读取源文件了解方法签名**

Run: `head -300 packages/markitai/src/markitai/providers/claude_agent.py`

**Step 2: 添加测试类**

在 `packages/markitai/tests/unit/test_providers.py` 中添加：

```python
class TestClaudeAgentProviderHelpers:
    """Tests for ClaudeAgentProvider helper methods."""

    @pytest.fixture
    def provider(self) -> Any:
        """Create a ClaudeAgentProvider instance for testing."""
        from markitai.providers.claude_agent import ClaudeAgentProvider, LITELLM_AVAILABLE

        if not LITELLM_AVAILABLE:
            pytest.skip("LiteLLM not available")
        return ClaudeAgentProvider()

    def test_has_images_with_image_url(self, provider: Any) -> None:
        """Test _has_images with image_url content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}
                ],
            }
        ]
        assert provider._has_images(messages) is True

    def test_has_images_text_only(self, provider: Any) -> None:
        """Test _has_images with text-only content."""
        messages = [{"role": "user", "content": "Hello"}]
        assert provider._has_images(messages) is False

    def test_has_images_mixed_content(self, provider: Any) -> None:
        """Test _has_images with mixed text and image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]
        assert provider._has_images(messages) is True

    def test_has_images_empty_messages(self, provider: Any) -> None:
        """Test _has_images with empty messages."""
        messages: list[dict[str, Any]] = []
        assert provider._has_images(messages) is False

    def test_has_images_multiple_messages_one_with_image(self, provider: Any) -> None:
        """Test _has_images with multiple messages, one containing image."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now look at this:"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xxx"}},
                ],
            },
        ]
        assert provider._has_images(messages) is True
```

**Step 3: 运行测试验证通过**

Run: `cd packages/markitai && uv run pytest tests/unit/test_providers.py::TestClaudeAgentProviderHelpers -v`
Expected: PASS

**Step 4: 提交**

```bash
git add packages/markitai/tests/unit/test_providers.py
git commit -m "test: add tests for ClaudeAgentProvider helper methods"
```

---

## Task 6: 测试 `_context_display_name()` 函数

**Files:**
- Modify: `packages/markitai/tests/unit/test_document_utils.py`
- Source: `packages/markitai/src/markitai/llm/document.py:35-52`

**Step 1: 添加测试类**

在 `packages/markitai/tests/unit/test_document_utils.py` 中添加：

```python
from markitai.llm.document import _context_display_name


class TestContextDisplayName:
    """Tests for _context_display_name function."""

    def test_empty_context(self) -> None:
        """Test with empty string."""
        assert _context_display_name("") == "unknown"

    def test_none_context(self) -> None:
        """Test with None-like input."""
        assert _context_display_name("") == "unknown"

    def test_file_path_unix(self) -> None:
        """Test with Unix file path."""
        assert _context_display_name("/home/user/docs/file.md") == "file.md"

    def test_file_path_windows(self) -> None:
        """Test with Windows file path."""
        assert _context_display_name("C:\\Users\\docs\\file.md") == "file.md"

    def test_short_url(self) -> None:
        """Test with short URL."""
        url = "https://example.com/page"
        assert _context_display_name(url) == url

    def test_long_url_truncated(self) -> None:
        """Test that long URLs are truncated."""
        long_url = "https://example.com/" + "a" * 50
        result = _context_display_name(long_url)

        assert len(result) == 50
        assert result.endswith("...")

    def test_simple_filename(self) -> None:
        """Test with simple filename (no path)."""
        assert _context_display_name("document.pdf") == "document.pdf"
```

**Step 2: 运行测试验证通过**

Run: `cd packages/markitai && uv run pytest tests/unit/test_document_utils.py::TestContextDisplayName -v`
Expected: PASS

**Step 3: 提交**

```bash
git add packages/markitai/tests/unit/test_document_utils.py
git commit -m "test: add tests for _context_display_name function"
```

---

## Task 7: 添加 CLI cache stats 命令测试

**Files:**
- Modify: `packages/markitai/tests/unit/test_cache_cli.py`
- Source: `packages/markitai/src/markitai/cli/commands/cache.py:42-159`

**Step 1: 添加 CLI 测试**

在 `packages/markitai/tests/unit/test_cache_cli.py` 中添加：

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from markitai.cli.commands.cache import cache_stats, cache_clear, cache_spa_domains


class TestCacheStatsCommand:
    """Tests for cache stats CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.cache.enabled = True
        config.cache.global_dir = str(tmp_path)
        config.cache.max_size_bytes = 100 * 1024 * 1024
        return config

    def test_stats_no_cache_exists(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats when no cache exists."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats)

            assert result.exit_code == 0
            assert "Cache Statistics" in result.output or "No cache found" in result.output

    def test_stats_json_format(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats with JSON output format."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats, ["--json"])

            assert result.exit_code == 0
            # Should be valid JSON
            data = json.loads(result.output)
            assert "enabled" in data
            assert data["enabled"] is True

    def test_stats_with_existing_cache(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats with existing cache."""
        from markitai.llm import SQLiteCache
        from markitai.constants import DEFAULT_CACHE_DB_FILENAME

        # Create a cache with some entries
        cache_path = tmp_path / DEFAULT_CACHE_DB_FILENAME
        cache = SQLiteCache(cache_path, mock_config.cache.max_size_bytes)
        cache.set("test_key", "test_content", "test_response", model="test-model")
        cache.close()

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats)

            assert result.exit_code == 0
            assert "Entries:" in result.output or "count" in result.output


class TestCacheClearCommand:
    """Tests for cache clear CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.cache.enabled = True
        config.cache.global_dir = str(tmp_path)
        config.cache.max_size_bytes = 100 * 1024 * 1024
        return config

    def test_clear_aborted_without_yes(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test clear is aborted when user doesn't confirm."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_clear, input="n\n")

            assert result.exit_code == 0
            assert "Aborted" in result.output

    def test_clear_with_yes_flag(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test clear with --yes flag skips confirmation."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_clear, ["--yes"])

            assert result.exit_code == 0
            # Either cleared something or nothing to clear
            assert "Cleared" in result.output or "No cache entries" in result.output


class TestCacheSpaDomainsCommand:
    """Tests for cache spa-domains CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_spa_domains_empty(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test spa-domains with no learned domains."""
        with patch("markitai.cli.commands.cache.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.list_domains.return_value = []
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains)

            assert result.exit_code == 0
            assert "No learned SPA domains" in result.output

    def test_spa_domains_json_format(self, runner: CliRunner) -> None:
        """Test spa-domains with JSON output."""
        with patch("markitai.cli.commands.cache.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.list_domains.return_value = [
                {"domain": "example.com", "hits": 5, "expired": False}
            ]
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) == 1
            assert data[0]["domain"] == "example.com"

    def test_spa_domains_clear(self, runner: CliRunner) -> None:
        """Test spa-domains --clear."""
        with patch("markitai.cli.commands.cache.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.clear.return_value = 3
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains, ["--clear"])

            assert result.exit_code == 0
            assert "Cleared 3" in result.output
            mock_cache.clear.assert_called_once()
```

**Step 2: 运行测试验证通过**

Run: `cd packages/markitai && uv run pytest tests/unit/test_cache_cli.py -v`
Expected: PASS

**Step 3: 提交**

```bash
git add packages/markitai/tests/unit/test_cache_cli.py
git commit -m "test: add CLI tests for cache stats, clear, and spa-domains commands"
```

---

## Task 8: 运行完整测试套件并验证覆盖率

**Files:**
- None (verification only)

**Step 1: 运行完整测试**

Run: `cd packages/markitai && uv run pytest --cov=markitai --cov-report=term-missing -v 2>&1 | tail -50`
Expected: All tests pass

**Step 2: 检查覆盖率提升**

Run: `cd packages/markitai && uv run pytest --cov=markitai --cov-report=html`
Then check `packages/markitai/htmlcov/index.html` for detailed coverage report.

**Step 3: 提交所有更改（如果尚未提交）**

```bash
git status
# If any uncommitted changes:
git add -A
git commit -m "test: complete phase 1 test coverage improvement"
```

---

## 预期成果

完成上述任务后：

| 模块 | 之前 | 之后 |
|------|------|------|
| `cli/commands/cache.py` | 15% | ~60% |
| `fetch.py` | 28% | ~40% |
| `llm/document.py` | 30% | ~45% |
| `providers/copilot.py` | 34% | ~50% |
| `providers/claude_agent.py` | 42% | ~55% |
| **总体覆盖率** | **53%** | **~60%** |

---

## 后续任务（第二阶段）

完成第一阶段后，下一阶段将包括：

1. **Task 9-10**: CLI config 命令测试
2. **Task 11-12**: FetchCache 集成测试
3. **Task 13-15**: LLM processor Mock 测试

这些任务需要更多 Mock 设置和外部依赖处理，建议在第一阶段验证通过后再继续。
