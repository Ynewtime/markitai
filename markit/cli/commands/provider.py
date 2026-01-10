"""Provider management commands for testing and listing models."""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import questionary
import typer
from rich.console import Console
from rich.table import Table

from markit.config import get_settings
from markit.utils.capabilities import infer_capabilities
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
    name: str | None = None  # User-defined name
    latency_ms: float | None = None
    models_count: int | None = None
    error: str | None = None
    model_configured: str | None = None


async def _test_openai(
    api_key: str,
    base_url: str | None = None,
    timeout: float = 10.0,
) -> ProviderTestResult:
    """Test OpenAI connectivity using models.list() endpoint."""
    from openai import AsyncOpenAI

    start = time.perf_counter()
    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
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


def _get_effective_api_key(config: Any) -> str | None:
    """Get API key from config, checking env vars if needed."""
    api_key = getattr(config, "api_key", None)
    if not api_key:
        api_key_env = getattr(config, "api_key_env", None)
        if api_key_env:
            api_key = os.environ.get(api_key_env)

    if not api_key:
        provider_name = config.provider
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var = env_vars.get(provider_name)
        if env_var:
            api_key = os.environ.get(env_var)
    return api_key


def _get_unique_credentials(
    settings,
) -> list[tuple[str, str | None, str | None, str, str | None]]:
    """Get unique credentials.

    Returns:
        List of (provider_name, api_key, base_url, display_name, credential_id)
    """
    creds = []
    seen = set()

    # 1. Process new credentials
    for cred in settings.llm.credentials:
        api_key = _get_effective_api_key(cred)
        key = (cred.provider, api_key, cred.base_url)
        if key not in seen:
            seen.add(key)
            creds.append((cred.provider, api_key, cred.base_url, cred.id, cred.id))

    # 2. Process legacy providers (de-duplicate)
    for config in settings.llm.providers:
        api_key = _get_effective_api_key(config)
        key = (config.provider, api_key, config.base_url)
        if key not in seen:
            seen.add(key)
            display_name = config.name or config.provider
            # Legacy providers don't have a credential ID
            creds.append((config.provider, api_key, config.base_url, display_name, None))

    return creds


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
    providers_to_test = _get_unique_credentials(settings)

    if show_progress:
        console.print(f"[cyan]Testing {len(providers_to_test)} unique credential(s)...[/cyan]")

    # Test each provider
    for provider_name, api_key, base_url, display_name, _ in providers_to_test:
        if show_progress:
            console.print(f"  Testing [bold]{display_name}[/bold]...", end=" ")

        result: ProviderTestResult

        if provider_name == "openai":
            if not api_key:
                result = ProviderTestResult(
                    provider=provider_name,
                    status="skipped",
                    error="No API key configured",
                )
            else:
                result = await _test_openai(api_key, base_url=base_url)

        elif provider_name == "anthropic":
            if not api_key:
                result = ProviderTestResult(
                    provider=provider_name,
                    status="skipped",
                    error="No API key configured",
                )
            else:
                result = await _test_anthropic(api_key)

        elif provider_name == "gemini":
            if not api_key:
                result = ProviderTestResult(
                    provider=provider_name,
                    status="skipped",
                    error="No API key configured",
                )
            else:
                result = await _test_gemini(api_key)

        elif provider_name == "openrouter":
            if not api_key:
                result = ProviderTestResult(
                    provider=provider_name,
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

        result.name = display_name
        # Note: model_configured is not applicable for credential testing
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
    table.add_column("Name", style="cyan")  # Changed from "Provider" to "Name"
    table.add_column("Type", style="dim")  # Added "Type" column
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
            result.name or result.provider,
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


async def _get_openai_models(
    api_key: str,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Get OpenAI models list."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
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
        creds = _get_unique_credentials(settings)

        for provider_name, api_key, base_url, display_name, _ in creds:
            if (
                provider_filter
                and provider_name != provider_filter
                and display_name != provider_filter
            ):
                continue

            console.print(f"Fetching models from [cyan]{display_name}[/cyan]...", end=" ")

            try:
                if provider_name == "openai" and api_key:
                    models = await _get_openai_models(api_key, base_url=base_url)
                elif provider_name == "anthropic" and api_key:
                    models = await _get_anthropic_models(api_key)
                elif provider_name == "gemini" and api_key:
                    models = await _get_gemini_models(api_key)
                elif provider_name == "openrouter" and api_key:
                    models = await _get_openrouter_models(api_key)
                elif provider_name == "ollama":
                    host = base_url or "http://localhost:11434"
                    models = await _get_ollama_models(host)
                else:
                    console.print("[yellow]skipped (no API key)[/yellow]")
                    continue

                # Use display_name as key to support multiple providers of same type
                all_models[display_name] = models
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


def _get_models_sync(
    provider_name: str, api_key: str | None, base_url: str | None
) -> list[dict[str, Any]]:
    """Get models list using synchronous API calls to avoid event loop issues."""
    models = []

    if provider_name == "openai" and api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        result = client.models.list()
        models = [{"id": m.id, "created": m.created, "owned_by": m.owned_by} for m in result.data]

    elif provider_name == "anthropic" and api_key:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key, timeout=30.0)
        result = client.models.list()
        models = [
            {
                "id": m.id,
                "display_name": m.display_name,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in result.data
        ]

    elif provider_name == "gemini" and api_key:
        from google import genai

        client = genai.Client(api_key=api_key)
        result = list(client.models.list())
        models = [
            {
                "id": m.name,
                "display_name": m.display_name,
                "description": m.description[:100] if m.description else None,
            }
            for m in result
        ]

    elif provider_name == "openrouter" and api_key:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30.0
        )
        result = client.models.list()
        models = [{"id": m.id, "created": m.created, "owned_by": m.owned_by} for m in result.data]

    elif provider_name == "ollama":
        import httpx

        host = base_url or "http://localhost:11434"
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{host}/api/tags")
            data = response.json()
            models = [
                {"id": m.get("name"), "size": m.get("size"), "modified_at": m.get("modified_at")}
                for m in data.get("models", [])
            ]

    return models


@provider_app.command("select")
def select_model() -> None:
    """Interactive wizard to select and configure a model.

    Allows you to:
    1. Select an existing provider/credential
    2. Browse available models (fetched live)
    3. Add the selected model to your markit.toml
    """
    settings = get_settings()
    creds = _get_unique_credentials(settings)

    if not creds:
        console.print("[red]No providers configured![/red]")
        console.print("Please configure at least one provider in markit.toml first.")
        raise typer.Exit(1)

    # 1. Select Credential
    choices = [
        questionary.Choice(
            title=f"{display_name} ({provider})",
            value=(provider, api_key, base_url, credential_id),
        )
        for provider, api_key, base_url, display_name, credential_id in creds
    ]

    selected = questionary.select(
        "Select a provider credential to use:",
        choices=choices,
    ).ask()

    if not selected:
        raise typer.Exit()

    provider_name, api_key, base_url, credential_id = selected

    # 2. Fetch Models using sync API to avoid event loop issues
    display_name = credential_id or provider_name
    console.print(f"Fetching models for [cyan]{display_name}[/cyan]...", end=" ")
    try:
        models = _get_models_sync(provider_name, api_key, base_url)

        if not models:
            console.print("[red]failed (no models found)[/red]")
            raise typer.Exit(1)

        console.print(f"[green]{len(models)} models found[/green]")

    except Exception as e:
        console.print(f"[red]failed ({e})[/red]")
        raise typer.Exit(1) from None

    # 3. Select Model with fuzzy search
    # Build model ID to display mapping
    model_id_to_display: dict[str, str] = {}
    model_display_list: list[str] = []

    for m in models:
        mid = m.get("id")
        if not mid:
            continue

        caps = infer_capabilities(mid)
        cap_str = f" [{', '.join(caps)}]" if caps != ["text"] else ""
        display = f"{mid}{cap_str}"
        model_id_to_display[mid] = display
        model_display_list.append(display)

    # Sort for easier browsing
    model_display_list.sort()

    # Use autocomplete with fuzzy matching and validation
    def validate_model(text: str) -> bool | str:
        """Validate that input matches a model."""
        if not text:
            return "Please select a model"
        # Check if it's a valid model ID or display string
        if text in model_display_list:
            return True
        # Check if it matches a model ID directly
        if text in model_id_to_display:
            return True
        return "Please select a model from the list"

    selected_model_display = questionary.autocomplete(
        "Select a model (type to search):",
        choices=model_display_list,
        match_middle=True,
        validate=validate_model,
    ).ask()

    if not selected_model_display:
        raise typer.Exit()

    # Extract real model ID from display string
    # Display format: "model-id [capabilities]" or just "model-id"
    real_model_id = selected_model_display.split(" [")[0].strip()

    if real_model_id not in model_id_to_display:
        console.print("[red]Invalid model selection[/red]")
        raise typer.Exit(1)

    # 4. Configure Name
    default_name = real_model_id
    display_name = questionary.text(
        "Enter a display name for this configuration:",
        default=default_name,
    ).ask()

    if display_name is None:
        # User cancelled with Ctrl+C
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit()

    # 5. Write to Config using manual text insertion for precise placement
    config_path = Path("markit.toml")
    if not config_path.exists():
        console.print("[red]markit.toml not found![/red]")
        raise typer.Exit(1)

    try:
        config_content = config_path.read_text(encoding="utf-8")
        caps = infer_capabilities(real_model_id)

        # Prepare capabilities string
        caps_line = ""
        if caps != ["text"]:
            caps_quoted = [f'"{c}"' for c in caps]
            caps_line = f'capabilities = [{", ".join(caps_quoted)}]\n'

        if credential_id:
            # New Schema
            section = "[[llm.models]]"
            new_block = (
                f"\n# Model: {display_name or real_model_id}\n"
                f"[[llm.models]]\n"
                f'name = "{display_name or real_model_id}"\n'
                f'model = "{real_model_id}"\n'
                f'credential_id = "{credential_id}"\n'
                f"{caps_line}"
            )
            target_header = "[[llm.models]]"
            section_marker = "# Models Configuration"
        else:
            # Legacy Schema
            section = "[[llm.providers]]"
            new_block = (
                f"\n# Provider: {display_name or provider_name}\n"
                f"[[llm.providers]]\n"
                f'provider = "{provider_name}"\n'
                f'model = "{real_model_id}"\n'
            )
            if display_name:
                new_block += f'name = "{display_name}"\n'
            if base_url:
                new_block += f'base_url = "{base_url}"\n'

            # API Key handling
            matching_config = None
            for conf in settings.llm.providers:
                if conf.provider == provider_name and conf.base_url == base_url:
                    matching_config = conf
                    break

            if matching_config:
                if matching_config.api_key_env:
                    new_block += f'api_key_env = "{matching_config.api_key_env}"\n'
                elif matching_config.api_key:
                    new_block += f'api_key = "{matching_config.api_key}"\n'

            new_block += f"{caps_line}"

            target_header = "[[llm.providers]]"
            section_marker = "# Legacy Schema"

        # Insertion Logic
        import re

        # 1. Try to find the last active entry of the target header
        header_pattern = re.compile(f"^{re.escape(target_header)}", re.MULTILINE)
        matches = list(header_pattern.finditer(config_content))

        insert_pos = -1

        if matches:
            last_header = matches[-1]
            # Find the start of the NEXT section or separator
            remaining_text = config_content[last_header.end():]

            # Look for next section header ([) or separator (# ===)
            next_section = re.search(r"^(\[|# ===)", remaining_text, re.MULTILINE)

            if next_section:
                insert_pos = last_header.end() + next_section.start()
            else:
                insert_pos = len(config_content)

        else:
            # No existing active entries. Find section marker.
            marker_match = re.search(re.escape(section_marker), config_content)
            if marker_match:
                # Find the end of the decorative block (usually followed by # ===)
                # We start searching from the marker match
                remaining = config_content[marker_match.end():]
                # Look for the separator line
                sep_match = re.search(r"^# =+", remaining, re.MULTILINE)

                if sep_match:
                    insert_pos = marker_match.end() + sep_match.end()
                else:
                    insert_pos = marker_match.end()
            else:
                # Fallback: Append to end
                insert_pos = len(config_content)

        # Perform insertion
        if insert_pos == len(config_content):
            config_content = config_content.rstrip() + "\n" + new_block
        else:
            # Insert and clean up newlines
            before = config_content[:insert_pos].rstrip()
            after = config_content[insert_pos:].lstrip("\n")
            config_content = f"{before}\n{new_block}\n{after}"

        config_path.write_text(config_content, encoding="utf-8")
        console.print(f"\n[green]Successfully added {real_model_id} to {section}![/green]")
        console.print("Run [bold]markit provider test[/bold] to verify.")

    except Exception as e:
        console.print(f"[red]Failed to update config: {e}[/red]")
        raise typer.Exit(1) from None
