#!/usr/bin/env python3
"""UAT: Test OpenRouter image analysis with JSON mode.

OpenRouter is an API aggregator that supports JSON mode for compatible models.

Usage:
    export OPENROUTER_API_KEY=your_key
    uv run python uat/test_openrouter_image.py [image_path]
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.image.analyzer import IMAGE_ANALYSIS_PROMPT

DEFAULT_IMAGE = "output/assets/Free_Test_Data_500KB_PPTX.pptx.001.png"


async def test_openrouter(image_path: Path):
    """Test OpenRouter image analysis with JSON mode."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        return False

    print("=" * 60)
    print("UAT: OpenRouter Image Analysis")
    print("Provider: OpenRouter (with JSON mode for supported models)")
    print(f"Image: {image_path.name}")
    print("=" * 60)

    image_data = image_path.read_bytes()
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"

    b64_image = base64.b64encode(image_data).decode("utf-8")

    print(f"Image size: {len(image_data):,} bytes")
    print()

    # OpenRouter uses OpenAI-compatible API
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Use a model that supports JSON mode
    model = "openai/gpt-4o"

    print(f"Model: {model}")
    print("JSON Mode: ✅ Enabled (response_format=json_object)")
    print("-" * 40)

    try:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            response_format={"type": "json_object"},  # JSON mode
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{b64_image}"},
                        },
                        {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    ],
                }
            ],
        )

        content = response.choices[0].message.content or ""

        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Output tokens: {response.usage.completion_tokens if response.usage else 'N/A'}")
        print()

        # With JSON mode, should parse directly
        try:
            data = json.loads(content)
            print("✅ JSON parsed successfully (native JSON mode)")
            print(f"   alt_text: {data.get('alt_text', 'N/A')[:50]}...")
            print(f"   image_type: {data.get('image_type', 'N/A')}")

            if content.strip().startswith("```"):
                print("⚠️  Unexpected: Response wrapped in code block despite JSON mode")

            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Response: {content[:200]}...")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    success = asyncio.run(test_openrouter(image_path))
    sys.exit(0 if success else 1)
