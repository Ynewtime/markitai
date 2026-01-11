#!/usr/bin/env python3
"""Run all UAT tests for image analysis across providers.

This script runs all provider-specific tests and summarizes results.

Usage:
    # Set all API keys first (optional - will skip missing)
    export ANTHROPIC_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    export GOOGLE_API_KEY=your_key
    export OPENROUTER_API_KEY=your_key

    # Run all tests
    uv run python uat/run_all.py [image_path]

    # Or run individual tests
    uv run python uat/test_anthropic_image.py
    uv run python uat/test_openai_image.py
    uv run python uat/test_gemini_image.py
    uv run python uat/test_ollama_image.py
    uv run python uat/test_openrouter_image.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_IMAGE = "output/assets/Free_Test_Data_500KB_PPTX.pptx.001.png"


async def run_all_tests(image_path: Path):
    """Run all provider tests and summarize results."""
    print("=" * 70)
    print("UAT: Image Analysis JSON Mode Tests")
    print("=" * 70)
    print(f"Image: {image_path}")
    print()

    results = {}

    # Check which providers are available
    providers = {
        "Anthropic": ("ANTHROPIC_API_KEY", "test_anthropic_image"),
        "OpenAI": ("OPENAI_API_KEY", "test_openai_image"),
        "Gemini": ("GOOGLE_API_KEY", "test_gemini_image"),
        "Ollama": (None, "test_ollama_image"),  # No API key needed
        "OpenRouter": ("OPENROUTER_API_KEY", "test_openrouter_image"),
    }

    print("Provider Availability:")
    print("-" * 40)
    for name, (env_var, _) in providers.items():
        if env_var is None:
            status = "✅ Local (check Ollama running)"
        elif os.environ.get(env_var):
            status = "✅ API key set"
        else:
            status = "⚠️  API key not set (will skip)"
        print(f"  {name}: {status}")
    print()

    # Run tests
    for name, (env_var, module_name) in providers.items():
        if env_var and not os.environ.get(env_var):
            results[name] = "SKIPPED"
            continue

        print()
        print("#" * 70)
        print(f"# Running: {name}")
        print("#" * 70)
        print()

        try:
            # Import and run the test
            module = __import__(module_name)
            test_func = getattr(
                module, f"test_{module_name.replace('test_', '').replace('_image', '')}"
            )
            success = await test_func(image_path)
            results[name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"❌ Test error: {e}")
            results[name] = "ERROR"

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    passed = sum(1 for v in results.values() if v == "PASSED")
    failed = sum(1 for v in results.values() if v == "FAILED")
    skipped = sum(1 for v in results.values() if v == "SKIPPED")
    errors = sum(1 for v in results.values() if v == "ERROR")

    for name, status in results.items():
        icon = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⏭️", "ERROR": "💥"}[status]
        json_mode = "Native JSON mode" if name != "Anthropic" else "Tool Use"
        print(f"  {icon} {name}: {status} ({json_mode})")

    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
    print()

    # JSON Mode comparison
    print("JSON Mode Support:")
    print("-" * 40)
    print("  ✅ OpenAI:     response_format={'type': 'json_object'}")
    print("  ✅ Gemini:     response_mime_type='application/json'")
    print("  ✅ Ollama:     format='json'")
    print("  ✅ OpenRouter: Depends on underlying model")
    print("  ✅ Anthropic:  Tool Use with tool_choice (structured output)")
    print()

    return failed == 0 and errors == 0


if __name__ == "__main__":
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    success = asyncio.run(run_all_tests(image_path))
    sys.exit(0 if success else 1)
