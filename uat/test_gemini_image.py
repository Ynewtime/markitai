#!/usr/bin/env python3
"""UAT: Test Google Gemini image analysis with native JSON mode.

Gemini supports native JSON mode via response_mime_type parameter.

Usage:
    export GOOGLE_API_KEY=your_key
    uv run python uat/test_gemini_image.py [image_path]
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.image.analyzer import IMAGE_ANALYSIS_PROMPT

DEFAULT_IMAGE = "output/assets/Free_Test_Data_500KB_PPTX.pptx.001.png"


async def test_gemini(image_path: Path):
    """Test Gemini image analysis with JSON mode."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set")
        return False

    print("=" * 60)
    print("UAT: Google Gemini Image Analysis")
    print("Provider: Google Gemini (with native JSON mode)")
    print(f"Image: {image_path.name}")
    print("=" * 60)

    image_data = image_path.read_bytes()
    image_format = image_path.suffix.lstrip(".").lower()
    if image_format == "jpg":
        image_format = "jpeg"

    print(f"Image size: {len(image_data):,} bytes")
    print()

    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    print(f"Model: {model}")
    print("JSON Mode: ✅ Enabled (response_mime_type=application/json)")
    print("-" * 40)

    try:
        # Create image part
        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type=f"image/{image_format}",
        )
        text_part = types.Part.from_text(text=IMAGE_ANALYSIS_PROMPT)

        # Enable JSON mode
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=[image_part, text_part],
            config=config,
        )

        content = response.text or ""

        print(
            f"Output tokens: {response.usage_metadata.candidates_token_count if response.usage_metadata else 'N/A'}"
        )
        print()

        # With JSON mode, should parse directly
        try:
            data = json.loads(content)
            print("✅ JSON parsed successfully (native JSON mode)")
            print(f"   alt_text: {data.get('alt_text', 'N/A')[:50]}...")
            print(f"   image_type: {data.get('image_type', 'N/A')}")

            # Check for code block (should NOT happen with JSON mode)
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
    success = asyncio.run(test_gemini(image_path))
    sys.exit(0 if success else 1)
