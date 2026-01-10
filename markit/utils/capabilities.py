"""Utility to infer capabilities of LLM models."""

import re

# Known vision-capable model patterns (lowercase)
VISION_PATTERNS = [
    r"gpt-4.*vision",
    r"gpt-4o",  # GPT-4o series
    r"gpt-4\.?5",  # GPT-4.5 series
    r"gpt-5",  # GPT-5 series
    r"claude-3",  # Claude 3 series
    r"claude-.*-4",  # Claude 4 series (claude-sonnet-4, claude-opus-4, etc.)
    r"gemini",  # Gemini series (embedding excluded separately)
    r"vision",
    r"llava",
    r"bakllava",
    r"yi-vl",
    r"qwen-vl",
]

# Known embedding models
EMBEDDING_PATTERNS = [
    r"embedding",
    r"bge-",
    r"e5-",
]


def infer_capabilities(model_id: str) -> list[str]:
    """Infer capabilities from model ID.

    Args:
        model_id: Model identifier string

    Returns:
        List of capabilities (e.g., ["text", "vision"])
    """
    model = model_id.lower()
    capabilities = ["text"]  # Assume all models support text generation

    # Check for vision capability
    for pattern in VISION_PATTERNS:
        if re.search(pattern, model):
            capabilities.append("vision")
            break

    # Check for embedding capability (usually mutually exclusive with text gen, but keeping simple)
    for pattern in EMBEDDING_PATTERNS:
        if re.search(pattern, model):
            if "text" in capabilities:
                capabilities.remove("text")
            capabilities.append("embedding")
            break

    return sorted(set(capabilities))
