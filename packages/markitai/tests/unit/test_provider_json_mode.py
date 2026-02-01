"""Unit tests for unified JSON mode handler.

These tests verify the StructuredOutputHandler class which provides unified
JSON handling for local LLM providers (claude-agent and copilot), eliminating
duplicate JSON parsing and prompt generation code.
"""

from __future__ import annotations


class TestCleanControlCharacters:
    """Tests for clean_control_characters function."""

    def test_removes_null_byte(self) -> None:
        """Test that null bytes are removed from text."""
        from markitai.providers.json_mode import clean_control_characters

        result = clean_control_characters("hello\x00world")
        assert result == "helloworld"

    def test_removes_various_control_characters(self) -> None:
        """Test that various ASCII control characters are removed."""
        from markitai.providers.json_mode import clean_control_characters

        # Characters 0x00-0x08, 0x0b, 0x0c, 0x0e-0x1f, 0x7f
        text = "hello\x01\x02\x03\x0b\x0c\x0e\x1f\x7fworld"
        result = clean_control_characters(text)
        assert result == "helloworld"

    def test_preserves_newline(self) -> None:
        """Test that newline character is preserved."""
        from markitai.providers.json_mode import clean_control_characters

        result = clean_control_characters("hello\nworld")
        assert result == "hello\nworld"

    def test_preserves_tab(self) -> None:
        """Test that tab character is preserved."""
        from markitai.providers.json_mode import clean_control_characters

        result = clean_control_characters("hello\tworld")
        assert result == "hello\tworld"

    def test_preserves_carriage_return(self) -> None:
        """Test that carriage return character is preserved."""
        from markitai.providers.json_mode import clean_control_characters

        result = clean_control_characters("hello\rworld")
        assert result == "hello\rworld"

    def test_preserves_normal_text(self) -> None:
        """Test that normal text is unchanged."""
        from markitai.providers.json_mode import clean_control_characters

        text = "Hello, World! 你好世界 🎉"
        result = clean_control_characters(text)
        assert result == text


class TestBuildJsonPromptSuffix:
    """Tests for StructuredOutputHandler.build_json_prompt_suffix."""

    def test_returns_json_instruction(self) -> None:
        """Test that build_json_prompt_suffix returns JSON instruction text."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        result = handler.build_json_prompt_suffix()
        assert "JSON" in result
        assert result.strip()  # Non-empty

    def test_with_schema_includes_schema(self) -> None:
        """Test that schema is included in the prompt suffix."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = handler.build_json_prompt_suffix(schema=schema)
        assert "JSON" in result
        assert "name" in result


class TestExtractJson:
    """Tests for StructuredOutputHandler.extract_json."""

    def test_extracts_plain_json_object(self) -> None:
        """Test extraction of plain JSON object."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        result = handler.extract_json('{"name": "test", "value": 42}')
        assert result == {"name": "test", "value": 42}

    def test_extracts_plain_json_array(self) -> None:
        """Test extraction of plain JSON array."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        result = handler.extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_extracts_from_markdown_code_block(self) -> None:
        """Test extraction from markdown code block."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        text = """Here is the JSON:
```json
{"key": "value"}
```
"""
        result = handler.extract_json(text)
        assert result == {"key": "value"}

    def test_extracts_from_response_with_surrounding_text(self) -> None:
        """Test extraction from response with surrounding text."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        text = 'Sure, here is the data:\n{"result": true}\nHope this helps!'
        result = handler.extract_json(text)
        assert result == {"result": True}

    def test_returns_none_for_invalid_json(self) -> None:
        """Test that None is returned for invalid JSON."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        result = handler.extract_json("not valid json at all")
        assert result is None

    def test_returns_none_for_malformed_json(self) -> None:
        """Test that None is returned for malformed JSON."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        result = handler.extract_json("{missing: quotes}")
        assert result is None

    def test_cleans_control_characters(self) -> None:
        """Test that control characters are cleaned before parsing."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        # JSON with embedded control characters
        text = '{"name": "test\x00value"}'
        result = handler.extract_json(text)
        assert result == {"name": "testvalue"}

    def test_extracts_from_code_block_without_language(self) -> None:
        """Test extraction from code block without language specifier."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        text = """```
{"data": 123}
```"""
        result = handler.extract_json(text)
        assert result == {"data": 123}


class TestGenerateJsonSystemPrompt:
    """Tests for StructuredOutputHandler.generate_json_system_prompt."""

    def test_adds_json_instructions_to_base_prompt(self) -> None:
        """Test that JSON instructions are added to base prompt."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        base = "You are a helpful assistant."
        result = handler.generate_json_system_prompt(base)
        assert "You are a helpful assistant." in result
        assert "JSON" in result

    def test_with_schema_includes_schema(self) -> None:
        """Test that schema is included in generated prompt."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        base = "Analyze the data."
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        result = handler.generate_json_system_prompt(base, schema=schema)
        assert "Analyze the data." in result
        assert "score" in result


class TestValidateJson:
    """Tests for StructuredOutputHandler.validate_json."""

    def test_valid_data_returns_true(self) -> None:
        """Test that valid data returns True."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {"name": "test"}
        result = handler.validate_json(data, schema)
        assert result is True

    def test_missing_required_field_returns_false(self) -> None:
        """Test that missing required field returns False."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {}  # Missing "name"
        result = handler.validate_json(data, schema)
        assert result is False

    def test_wrong_type_returns_false(self) -> None:
        """Test that wrong type returns False."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {"type": "array"}
        data = {"not": "array"}
        result = handler.validate_json(data, schema)
        assert result is False

    def test_valid_array_returns_true(self) -> None:
        """Test that valid array returns True."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {"type": "array"}
        data = [1, 2, 3]
        result = handler.validate_json(data, schema)
        assert result is True
