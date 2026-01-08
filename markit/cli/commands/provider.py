"""Provider management commands for testing and listing models."""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from markit.config import get_settings
from markit.utils.logging import get_logger

# Default cache directory for markit
CACHE_DIR = Path.home() / ".cache" / "markit"
MODELS_CACHE_FILE = "models.json"


def get_cache_dir() -> Path:
    """Get the cache directory, creating it if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_models_cache_path() -> Path:
    """Get the path to the models cache file."""
    return get_cache_dir() / MODELS_CACHE_FILE


provider_app = typer.Typer(
    name="provider",
    help="Manage and test LLM providers.",
    no_args_is_help=True,
)

console = Console()
log = get_logger(__name__)


@dataclass
class ProviderTestResult:
    """Result of a provider connectivity test."""

    provider: str
    status: str  # "connected", "failed", "skipped"
    latency_ms: float | None = None
    models_count: int | None = None
    error: str | None = None
    model_configured: str | None = None


async def _test_openai(api_key: str, timeout: float = 10.0) -> ProviderTestResult:
    """Test OpenAI connectivity using models.list() endpoint."""
    from openai import AsyncOpenAI

    start = time.perf_counter()
    try:
        client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        models = await client.models.list()
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="openai",
            status="connected",
            latency_ms=latency,
            models_count=len(models.data),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="openai",
            status="failed",
            latency_ms=latency,
            error=str(e),
        )


async def _test_anthropic(api_key: str, timeout: float = 10.0) -> ProviderTestResult:
    """Test Anthropic connectivity using models.list() endpoint."""
    from anthropic import AsyncAnthropic

    start = time.perf_counter()
    try:
        client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        models = await client.models.list()
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="anthropic",
            status="connected",
            latency_ms=latency,
            models_count=len(list(models.data)),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="anthropic",
            status="failed",
            latency_ms=latency,
            error=str(e),
        )


async def _test_gemini(api_key: str) -> ProviderTestResult:
    """Test Gemini connectivity using models.list() endpoint."""
    from google import genai

    start = time.perf_counter()
    try:
        client = genai.Client(api_key=api_key)
        # Use sync list() in thread to avoid blocking
        models = list(client.models.list())
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="gemini",
            status="connected",
            latency_ms=latency,
            models_count=len(models),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="gemini",
            status="failed",
            latency_ms=latency,
            error=str(e),
        )


async def _test_openrouter(api_key: str, timeout: float = 10.0) -> ProviderTestResult:
    """Test OpenRouter connectivity using models endpoint."""
    from openai import AsyncOpenAI

    start = time.perf_counter()
    try:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        )
        models = await client.models.list()
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="openrouter",
            status="connected",
            latency_ms=latency,
            models_count=len(models.data),
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="openrouter",
            status="failed",
            latency_ms=latency,
            error=str(e),
        )


async def _test_ollama(host: str = "http://localhost:11434") -> ProviderTestResult:
    """Test Ollama connectivity using health endpoint."""
    import httpx

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # First check health endpoint
            response = await client.get(host)
            if response.status_code != 200 or "Ollama is running" not in response.text:
                raise Exception("Ollama health check failed")

            # Then get models list
            tags_response = await client.get(f"{host}/api/tags")
            tags_response.raise_for_status()
            tags_data = tags_response.json()
            latency = (time.perf_counter() - start) * 1000

            return ProviderTestResult(
                provider="ollama",
                status="connected",
                latency_ms=latency,
                models_count=len(tags_data.get("models", [])),
            )
    except httpx.ConnectError:
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="ollama",
            status="failed",
            latency_ms=latency,
            error="Connection refused - Is Ollama running? (ollama serve)",
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ProviderTestResult(
            provider="ollama",
            status="failed",
            latency_ms=latency,
            error=str(e),
        )


async def test_all_providers(
    settings=None,
    show_progress: bool = True,
) -> list[ProviderTestResult]:
    """Test all configured providers.

    Args:
        settings: MarkIt settings (uses default if None)
        show_progress: Show progress messages

    Returns:
        List of test results for each provider
    """
    if settings is None:
        settings = get_settings()

    results: list[ProviderTestResult] = []
    providers_to_test = []

    # Collect configured providers
    for config in settings.llm.providers:
        provider_name = config.provider
        api_key = config.api_key

        # Try to get API key from environment if not in config
        if not api_key:
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GOOGLE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            env_var = env_vars.get(provider_name)
            if env_var:
                api_key = os.environ.get(env_var)

        providers_to_test.append((provider_name, api_key, config.model, config.base_url))

    if show_progress:
        console.print(f"[cyan]Testing {len(providers_to_test)} configured provider(s)...[/cyan]")

    # Test each provider
    for provider_name, api_key, model, base_url in providers_to_test:
        if show_progress:
            console.print(f"  Testing [bold]{provider_name}[/bold]...", end=" ")

        result: ProviderTestResult

        if provider_name == "openai":
            if not api_key:
                result = ProviderTestResult(
                    provider="openai",
                    status="skipped",
                    error="No API key configured",
                )
            else:
                result = await _test_openai(api_key)

        elif provider_name == "anthropic":
            if not api_key:
                result = ProviderTestResult(
                    provider="anthropic",
                    status="skipped",
                    error="No API key configured",
                )
            else:
                result = await _test_anthropic(api_key)

        elif provider_name == "gemini":
            if not api_key:
                result = ProviderTestResult(
                    provider="gemini",
                    status="skipped",
                    error="No API key configured",
                )
            else:
                result = await _test_gemini(api_key)

        elif provider_name == "openrouter":
            if not api_key:
                result = ProviderTestResult(
                    provider="openrouter",
                    status="skipped",
                    error="No API key configured",
                )
            else:
                result = await _test_openrouter(api_key)

        elif provider_name == "ollama":
            host = base_url or "http://localhost:11434"
            result = await _test_ollama(host)

        else:
            result = ProviderTestResult(
                provider=provider_name,
                status="skipped",
                error=f"Unknown provider: {provider_name}",
            )

        result.model_configured = model
        results.append(result)

        if show_progress:
            if result.status == "connected":
                console.print(f"[green]OK[/green] ({result.latency_ms:.0f}ms)")
            elif result.status == "skipped":
                console.print(f"[yellow]SKIPPED[/yellow] ({result.error})")
            else:
                console.print(f"[red]FAILED[/red] ({result.error})")

    return results


def display_test_results(results: list[ProviderTestResult]) -> None:
    """Display test results in a table."""
    table = Table(title="Provider Test Results")
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Models", justify="right")
    table.add_column("Configured Model")
    table.add_column("Error")

    for result in results:
        status_style = {
            "connected": "[green]Connected[/green]",
            "failed": "[red]Failed[/red]",
            "skipped": "[yellow]Skipped[/yellow]",
        }.get(result.status, result.status)

        latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "-"
        models = str(result.models_count) if result.models_count is not None else "-"
        error = (
            result.error[:50] + "..."
            if result.error and len(result.error) > 50
            else (result.error or "-")
        )

        table.add_row(
            result.provider,
            status_style,
            latency,
            models,
            result.model_configured or "-",
            error,
        )

    console.print(table)


@provider_app.command("test")
def test_providers(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed output.",
        ),
    ] = False,
) -> None:
    """Test connectivity to all configured LLM providers.

    Tests each provider configured in markit.toml by calling their
    models endpoint (no tokens consumed).
    """
    from markit.utils.logging import setup_logging

    if verbose:
        setup_logging(level="DEBUG")

    try:
        results = asyncio.run(test_all_providers(show_progress=True))
        console.print()
        display_test_results(results)

        # Summary
        connected = sum(1 for r in results if r.status == "connected")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")

        console.print()
        if failed > 0:
            console.print(
                f"[yellow]Summary:[/yellow] {connected} connected, {failed} failed, {skipped} skipped"
            )
            raise typer.Exit(1)
        else:
            console.print(f"[green]Summary:[/green] {connected} connected, {skipped} skipped")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


async def _get_openai_models(api_key: str) -> list[dict[str, Any]]:
    """Get OpenAI models list."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, timeout=30.0)
    models = await client.models.list()
    return [
        {
            "id": m.id,
            "created": m.created,
            "owned_by": m.owned_by,
        }
        for m in models.data
    ]


async def _get_anthropic_models(api_key: str) -> list[dict[str, Any]]:
    """Get Anthropic models list."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key, timeout=30.0)
    models = await client.models.list()
    return [
        {
            "id": m.id,
            "display_name": m.display_name,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in models.data
    ]


async def _get_gemini_models(api_key: str) -> list[dict[str, Any]]:
    """Get Gemini models list."""
    from google import genai

    client = genai.Client(api_key=api_key)
    models = list(client.models.list())
    return [
        {
            "id": m.name,
            "display_name": m.display_name,
            "description": m.description[:100] if m.description else None,
        }
        for m in models
    ]


async def _get_openrouter_models(api_key: str) -> list[dict[str, Any]]:
    """Get OpenRouter models list."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30.0,
    )
    models = await client.models.list()
    return [
        {
            "id": m.id,
            "created": m.created,
            "owned_by": m.owned_by,
        }
        for m in models.data
    ]


async def _get_ollama_models(host: str) -> list[dict[str, Any]]:
    """Get Ollama models list."""
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{host}/api/tags")
        data = response.json()
        return [
            {
                "id": m.get("name"),
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
            }
            for m in data.get("models", [])
        ]


@provider_app.command("models")
def list_models(
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Don't save models to local cache.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Custom output file path (default: ~/.cache/markit/models.json).",
        ),
    ] = None,
    provider_filter: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="Filter by provider name.",
        ),
    ] = None,
) -> None:
    """List available models from all configured providers.

    Fetches the models list from each provider's API and saves
    to local cache (~/.cache/markit/models.json) by default.
    """
    settings = get_settings()
    all_models: dict[str, list[dict[str, Any]]] = {}

    async def fetch_all_models() -> None:
        for config in settings.llm.providers:
            provider_name = config.provider

            if provider_filter and provider_name != provider_filter:
                continue

            api_key = config.api_key
            if not api_key:
                env_vars = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "gemini": "GOOGLE_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY",
                }
                env_var = env_vars.get(provider_name)
                if env_var:
                    api_key = os.environ.get(env_var)

            console.print(f"Fetching models from [cyan]{provider_name}[/cyan]...", end=" ")

            try:
                if provider_name == "openai" and api_key:
                    models = await _get_openai_models(api_key)
                elif provider_name == "anthropic" and api_key:
                    models = await _get_anthropic_models(api_key)
                elif provider_name == "gemini" and api_key:
                    models = await _get_gemini_models(api_key)
                elif provider_name == "openrouter" and api_key:
                    models = await _get_openrouter_models(api_key)
                elif provider_name == "ollama":
                    host = config.base_url or "http://localhost:11434"
                    models = await _get_ollama_models(host)
                else:
                    console.print("[yellow]skipped (no API key)[/yellow]")
                    continue

                all_models[provider_name] = models
                console.print(f"[green]{len(models)} models[/green]")

            except Exception as e:
                console.print(f"[red]failed ({e})[/red]")

    asyncio.run(fetch_all_models())

    # Display models
    for provider_name, models in all_models.items():
        console.print()
        table = Table(title=f"{provider_name.upper()} Models ({len(models)})")
        table.add_column("Model ID", style="cyan")
        table.add_column("Info")

        # Show first 20 models
        for model in models[:20]:
            model_id = model.get("id", "unknown")
            info_parts = []
            if model.get("display_name"):
                info_parts.append(model["display_name"])
            if model.get("owned_by"):
                info_parts.append(f"by {model['owned_by']}")
            if model.get("size"):
                size_gb = model["size"] / (1024**3)
                info_parts.append(f"{size_gb:.1f}GB")
            info = ", ".join(info_parts) if info_parts else "-"
            table.add_row(model_id, info)

        if len(models) > 20:
            table.add_row("...", f"({len(models) - 20} more)")

        console.print(table)

    # Always save to local cache unless --no-cache is specified
    if not no_cache and all_models:
        cache_file = output or get_models_cache_path()
        cache_data = {
            "fetched_at": datetime.now().isoformat(),
            "providers": all_models,
        }
        # Ensure parent directory exists for custom output path
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache_data, indent=2, default=str))
        console.print(f"\n[green]Models cached to {cache_file}[/green]")
