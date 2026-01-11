#!/usr/bin/env python3
"""UAT: Test Ollama image analysis with native JSON mode.

Ollama supports native JSON mode via format parameter.

Usage:
    # Make sure Ollama is running locally with a vision model
    ollama pull llama3.2-vision

    uv run python uat/test_ollama_image.py [image_path]
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from markit.image.analyzer import IMAGE_ANALYSIS_PROMPT

DEFAULT_IMAGE = "output/assets/Free_Test_Data_500KB_PPTX.pptx.001.png"


async def test_ollama(image_path: Path):
    """Test Ollama image analysis with JSON mode."""
    import ollama

    print("=" * 60)
    print("UAT: Ollama Image Analysis")
    print("Provider: Ollama (local, with native JSON mode)")
    print(f"Image: {image_path.name}")
    print("=" * 60)

    image_data = image_path.read_bytes()
    b64_image = base64.b64encode(image_data).decode("utf-8")

    print(f"Image size: {len(image_data):,} bytes")
    print()

    client = ollama.AsyncClient()
    model = "llama3.2-vision"

    # Check if model is available
    try:
        models_response = await client.list()
        available_models = [m.model for m in models_response.models]
        if not any(model in m for m in available_models):
            print(f"❌ Model {model} not found")
            print(f"   Available: {available_models}")
            print(f"   Run: ollama pull {model}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False

    print(f"Model: {model}")
    print("JSON Mode: ✅ Enabled (format='json')")
    print("-" * 40)

    try:
        response = await client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": IMAGE_ANALYSIS_PROMPT,
                    "images": [b64_image],
                }
            ],
            format="json",  # Native JSON mode
        )

        content = response.get("message", {}).get("content", "")

        print(f"Done reason: {response.get('done_reason', 'N/A')}")
        print(f"Output tokens: {response.get('eval_count', 'N/A')}")
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
    success = asyncio.run(test_ollama(image_path))
    sys.exit(0 if success else 1)
