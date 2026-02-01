"""Unified JSON mode handler for structured LLM outputs.

This module provides a centralized handler for JSON mode operations across
local LLM providers (claude-agent, copilot), eliminating duplicate code for:
- JSON extraction from LLM responses
- JSON prompt suffix generation
- Control character cleaning
- Basic JSON schema validation

Usage:
    from markitai.providers.json_mode import StructuredOutputHandler

    handler = StructuredOutputHandler()

    # Generate system prompt with JSON instructions
    system_prompt = handler.generate_json_system_prompt(
        "You are a helpful assistant.",
        schema={"type": "object", "properties": {"name": {"type": "string"}}}
    )

    # Extract JSON from LLM response
    response_text = '{"name": "test"}'
    data = handler.extract_json(response_text)

    # Validate extracted data
    if data and handler.validate_json(data, schema):
        print("Valid response:", data)
"""

from __future__ import annotations

import json
import re
from typing import Any


def clean_control_characters(text: str) -> str:
    """Remove control characters that break JSON parsing.

    Removes ASCII control characters (0x00-0x08, 0x0b, 0x0c, 0x0e-0x1f, 0x7f)
    while preserving common whitespace characters (newline, tab, carriage return).

    Args:
        text: Input text that may contain control characters

    Returns:
        Text with control characters removed
    """
    # Remove control characters except:
    # - 0x09 (tab)
    # - 0x0a (newline)
    # - 0x0d (carriage return)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


class StructuredOutputHandler:
    """Handler for structured JSON outputs from LLM providers.

    This class provides methods for:
    - Building JSON instruction suffixes for prompts
    - Extracting JSON from LLM responses (including markdown code blocks)
    - Generating enhanced system prompts with JSON instructions
    - Basic JSON schema validation

    Attributes:
        _JSON_BLOCK_PATTERN: Regex to find ```json blocks in text
        _JSON_OBJECT_PATTERN: Regex to find JSON objects in text
    """

    # Pattern to match ```json ... ``` or ``` ... ``` code blocks
    _JSON_BLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*\n?([\s\S]*?)\n?```",
        re.IGNORECASE,
    )

    # Pattern to find JSON objects or arrays in text
    _JSON_OBJECT_PATTERN = re.compile(
        r"(\{[\s\S]*\}|\[[\s\S]*\])",
    )

    def build_json_prompt_suffix(self, schema: dict[str, Any] | None = None) -> str:
        """Build instruction suffix for JSON output.

        Generates text to append to prompts that instructs the LLM to respond
        with valid JSON. Optionally includes a JSON schema for the expected format.

        Args:
            schema: Optional JSON schema dict describing expected output format

        Returns:
            Instruction text to append to prompts
        """
        if schema:
            schema_str = json.dumps(schema, indent=2)
            return (
                "\n\nRespond with valid JSON only. "
                f"Your response must match this schema:\n```json\n{schema_str}\n```"
            )
        return "\n\nRespond with valid JSON only. Do not include any other text."

    def extract_json(self, text: str) -> dict[str, Any] | list[Any] | None:
        """Extract JSON from LLM response text.

        Handles various response formats:
        - Plain JSON objects or arrays
        - JSON wrapped in markdown code blocks (```json ... ```)
        - JSON embedded in explanatory text

        Control characters that would break JSON parsing are cleaned before extraction.

        Args:
            text: LLM response text that may contain JSON

        Returns:
            Parsed JSON as dict or list, or None if no valid JSON found
        """
        # Clean control characters first
        text = clean_control_characters(text)

        # Try to find JSON in markdown code block first
        block_match = self._JSON_BLOCK_PATTERN.search(text)
        if block_match:
            json_text = block_match.group(1).strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass  # Fall through to other methods

        # Try to parse the entire text as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object or array in the text
        obj_match = self._JSON_OBJECT_PATTERN.search(text)
        if obj_match:
            json_text = obj_match.group(1)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

        return None

    def generate_json_system_prompt(
        self,
        base_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate enhanced system prompt with JSON instructions.

        Combines the base system prompt with JSON output instructions.
        If a schema is provided, it is included to guide the LLM's response format.

        Args:
            base_prompt: Original system prompt
            schema: Optional JSON schema describing expected output format

        Returns:
            Enhanced system prompt with JSON instructions appended
        """
        suffix = self.build_json_prompt_suffix(schema)
        return base_prompt + suffix

    def validate_json(
        self,
        data: dict[str, Any] | list[Any],
        schema: dict[str, Any],
    ) -> bool:
        """Perform basic JSON schema validation.

        This provides simple validation for common schema patterns without
        requiring external validation libraries. It checks:
        - Type constraints (object, array, string, number, etc.)
        - Required fields for objects

        For full JSON Schema validation, consider using jsonschema library.

        Args:
            data: Parsed JSON data to validate
            schema: JSON schema dict to validate against

        Returns:
            True if data matches schema, False otherwise
        """
        schema_type = schema.get("type")

        # Validate type
        if schema_type:
            if schema_type == "object" and not isinstance(data, dict):
                return False
            if schema_type == "array" and not isinstance(data, list):
                return False
            if schema_type == "string" and not isinstance(data, str):
                return False
            if schema_type == "number" and not isinstance(data, (int, float)):
                return False
            if schema_type == "integer" and not isinstance(data, int):
                return False
            if schema_type == "boolean" and not isinstance(data, bool):
                return False

        # Validate required fields for objects
        if isinstance(data, dict):
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    return False

        return True


__all__ = [
    "clean_control_characters",
    "StructuredOutputHandler",
]
