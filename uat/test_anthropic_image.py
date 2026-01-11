#!/usr/bin/env python3
"""UAT: Test Anthropic image analysis with Tool Use for structured JSON output.

Anthropic does NOT support native JSON mode, but Tool Use with tool_choice
can be used to force structured output matching a JSON Schema.

Usage:
    export ANTHROPIC_API_KEY=your_key
    uv run python uat/test_anthropic_image.py [image_path]
"""

import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.image.analyzer import IMAGE_ANALYSIS_PROMPT
from markit.llm.base import IMAGE_ANALYSIS_SCHEMA

DEFAULT_IMAGE = "output/assets/Free_Test_Data_500KB_PPTX.pptx.001.png"


async def test_anthropic_tool_use(image_path: Path):
    """Test Anthropic image analysis with Tool Use (recommended approach)."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False

    print("=" * 60)
    print("UAT: Anthropic Image Analysis (Tool Use Mode)")
    print("Provider: Anthropic Claude")
    print(f"Image: {image_path.name}")
    print("=" * 60)

    image_data = image_path.read_bytes()
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"

    b64_image = base64.b64encode(image_data).decode("utf-8")

    print(f"Image size: {len(image_data):,} bytes")
    print()

    client = anthropic.AsyncAnthropic(api_key=api_key)
    model = "claude-sonnet-4-5-20250929"

    # Define tool for structured output
    tool = {
        "name": "output_image_analysis",
        "description": "Output structured image analysis results. Always use this tool.",
        "input_schema": IMAGE_ANALYSIS_SCHEMA,
    }

    print(f"Model: {model}")
    print("JSON Mode: ✅ Tool Use (tool_choice forces structured output)")
    print("-" * 40)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": "output_image_analysis"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    ],
                }
            ],
        )

        print(f"Stop reason: {response.stop_reason}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print()

        # Extract from tool_use block
        data = None
        for block in response.content:
            if block.type == "tool_use":
                data = block.input  # Already a dict
                break

        if data:
            print("✅ Structured output extracted from tool_use block")
            print(f"   alt_text: {data.get('alt_text', 'N/A')[:50]}...")
            print(f"   image_type: {data.get('image_type', 'N/A')}")

            # Verify it's valid JSON when serialized
            json_str = json.dumps(data, ensure_ascii=False)
            json.loads(json_str)  # Validate round-trip
            print("✅ JSON serialization verified")
            return True
        else:
            print("❌ No tool_use block in response")
            print(f"Response blocks: {[b.type for b in response.content]}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_anthropic_prompt(image_path: Path):
    """Test Anthropic image analysis with prompt engineering (fallback approach)."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False

    print()
    print("=" * 60)
    print("UAT: Anthropic Image Analysis (Prompt Mode - Fallback)")
    print("Provider: Anthropic Claude")
    print(f"Image: {image_path.name}")
    print("=" * 60)

    image_data = image_path.read_bytes()
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"

    b64_image = base64.b64encode(image_data).decode("utf-8")

    print(f"Image size: {len(image_data):,} bytes")
    print()

    client = anthropic.AsyncAnthropic(api_key=api_key)
    model = "claude-sonnet-4-5-20250929"

    print(f"Model: {model}")
    print("JSON Mode: ❌ Not supported (using prompt engineering)")
    print("-" * 40)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    ],
                }
            ],
        )

        content = response.content[0].text if response.content else ""

        print(f"Stop reason: {response.stop_reason}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print()

        # Check for code block wrapper
        has_code_block = content.strip().startswith("```")
        if has_code_block:
            print("⚠️  Response wrapped in code block (expected without JSON mode)")

        # Try to parse
        try:
            data = json.loads(content)
            print("✅ JSON parsed successfully (raw)")
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
            if match:
                data = json.loads(match.group(1))
                print("✅ JSON parsed successfully (from code block)")
            else:
                print("❌ Failed to parse JSON")
                print(f"Response: {content[:200]}...")
                return False

        print(f"   alt_text: {data.get('alt_text', 'N/A')[:50]}...")
        print(f"   image_type: {data.get('image_type', 'N/A')}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_anthropic(image_path: Path):
    """Run both Tool Use and Prompt tests."""
    # Test Tool Use mode (recommended)
    tool_success = await test_anthropic_tool_use(image_path)

    # Test Prompt mode (fallback)
    prompt_success = await test_anthropic_prompt(image_path)

    print()
    print("=" * 60)
    print("ANTHROPIC TEST SUMMARY")
    print("=" * 60)
    print(f"  Tool Use mode:   {'✅ PASSED' if tool_success else '❌ FAILED'}")
    print(f"  Prompt mode:     {'✅ PASSED' if prompt_success else '❌ FAILED'}")
    print()
    print("Recommendation: Use Tool Use for reliable structured output")
    print("=" * 60)

    return tool_success  # Primary success criterion


if __name__ == "__main__":
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    success = asyncio.run(test_anthropic(image_path))
    sys.exit(0 if success else 1)
