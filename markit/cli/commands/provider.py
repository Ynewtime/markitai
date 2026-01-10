"""Provider management commands for testing and listing credentials."""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import questionary
import typer
from rich.console import Console
from rich.table import Table

from markit.cli.shared import get_unique_credentials
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
    help="Manage and test LLM provider credentials.",
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


# Supported provider types with their default configurations
PROVIDER_TYPES = {
    "openai": {
        "display": "OpenAI",
        "default_api_key_env": "OPENAI_API_KEY",
        "base_url": None,  # Uses SDK default
        "requires_base_url": False,
    },
    "openai_compatible": {
        "display": "OpenAI Compatible (DeepSeek, Groq, etc.)",
        "default_api_key_env": None,  # User must specify
        "base_url": None,
        "requires_base_url": True,  # Must provide base_url
        "actual_provider": "openai",  # Stored as openai in config
    },
    "anthropic": {
        "display": "Anthropic",
        "default_api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
        "requires_base_url": False,
    },
    "gemini": {
        "display": "Google Gemini",
        "default_api_key_env": "GOOGLE_API_KEY",
        "base_url": None,
        "requires_base_url": False,
    },
    "openrouter": {
        "display": "OpenRouter",
        "default_api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "requires_base_url": False,
    },
    "ollama": {
        "display": "Ollama (Local)",
        "default_api_key_env": None,  # No API key needed
        "base_url": "http://localhost:11434",
        "requires_base_url": False,
    },
}


def _write_credential_to_config(credential: dict[str, Any]) -> bool:
    """Write a credential to markit.yaml using ruamel.yaml.

    Args:
        credential: Dict with id, provider, base_url, api_key_env fields

    Returns:
        True if successful, False otherwise
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    config_path = Path("markit.yaml")
    if not config_path.exists():
        console.print("[red]markit.yaml not found![/red]")
        console.print("Run [bold]markit config init[/bold] to create one first.")
        return False

    try:
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)

        config_data = yaml.load(config_path)

        # Ensure config_data is a valid map
        if config_data is None:
            config_data = CommentedMap()

        # Check if llm section exists and is valid
        llm_exists = "llm" in config_data and config_data["llm"] is not None

        if not llm_exists:
            # Need to create llm section
            llm_section = CommentedMap()
            llm_section["credentials"] = CommentedSeq()

            # Insert llm at the end - it's the safest approach to preserve
            # file structure and comments. The key order in YAML doesn't
            # affect functionality.
            config_data["llm"] = llm_section

            # Add a blank line before llm section for readability
            config_data.yaml_set_comment_before_after_key("llm", before="\n")
        else:
            llm_section = config_data["llm"]

            # Ensure credentials list exists
            if "credentials" not in llm_section or llm_section["credentials"] is None:
                llm_section["credentials"] = CommentedSeq()

        llm_section = config_data["llm"]

        # Check for duplicate credential ID
        for cred in llm_section["credentials"]:
            if cred.get("id") == credential["id"]:
                console.print(
                    f"[yellow]Credential with id '{credential['id']}' already exists.[/yellow]"
                )
                return False

        # Build the new credential entry
        new_cred = CommentedMap()
        new_cred["id"] = credential["id"]
        new_cred["provider"] = credential["provider"]

        if credential.get("base_url"):
            new_cred["base_url"] = credential["base_url"]

        if credential.get("api_key_env"):
            new_cred["api_key_env"] = credential["api_key_env"]

        # Append to credentials list
        llm_section["credentials"].append(new_cred)

        # Write back to file
        yaml.dump(config_data, config_path)
        return True

    except Exception as e:
        console.print(f"[red]Failed to write config: {e}[/red]")
        return False


def _interactive_add_credential() -> dict[str, Any] | None:
    """Interactive wizard to collect credential information.

    Returns:
        Dict with credential info, or None if cancelled
    """
    # 1. Select provider type
    provider_choices = [
        questionary.Choice(title=info["display"], value=ptype)
        for ptype, info in PROVIDER_TYPES.items()
    ]

    provider_type = questionary.select(
        "Select provider type:",
        choices=provider_choices,
    ).ask()

    if not provider_type:
        return None

    provider_info = PROVIDER_TYPES[provider_type]
    actual_provider = provider_info.get("actual_provider", provider_type)

    # 2. Enter credential ID
    default_id = provider_type.replace("_compatible", "")
    cred_id = questionary.text(
        "Enter a unique ID for this credential:",
        default=default_id,
        validate=lambda x: True if x.strip() else "ID cannot be empty",
    ).ask()

    if cred_id is None:
        return None

    cred_id = cred_id.strip()

    # 3. Base URL (required for openai_compatible, optional for others)
    base_url = None
    if provider_info["requires_base_url"]:
        base_url = questionary.text(
            "Enter the API base URL (required):",
            validate=lambda x: True if x.strip() else "Base URL is required for this provider",
        ).ask()
        if base_url is None:
            return None
        base_url = base_url.strip()
    elif provider_info.get("base_url"):
        # Has a default, ask if they want to change it
        use_default = questionary.confirm(
            f"Use default base URL ({provider_info['base_url']})?",
            default=True,
        ).ask()
        if use_default is None:
            return None
        if use_default:
            base_url = provider_info["base_url"]
        else:
            base_url = questionary.text(
                "Enter custom base URL:",
                validate=lambda x: True if x.strip() else "Base URL cannot be empty",
            ).ask()
            if base_url is None:
                return None
            base_url = base_url.strip()
    else:
        # No default, ask if they want to set one (optional)
        set_base_url = questionary.confirm(
            "Set a custom base URL? (optional)",
            default=False,
        ).ask()
        if set_base_url is None:
            return None
        if set_base_url:
            base_url = questionary.text(
                "Enter base URL:",
            ).ask()
            if base_url is None:
                return None
            base_url = base_url.strip() if base_url else None

    # 4. API key environment variable (not needed for Ollama)
    api_key_env = None
    if provider_type != "ollama":
        default_env = provider_info.get("default_api_key_env")
        if default_env:
            use_default_env = questionary.confirm(
                f"Use default API key env var ({default_env})?",
                default=True,
            ).ask()
            if use_default_env is None:
                return None
            if use_default_env:
                api_key_env = default_env
            else:
                api_key_env = questionary.text(
                    "Enter custom API key environment variable name:",
                    validate=lambda x: True if x.strip() else "Env var name cannot be empty",
                ).ask()
                if api_key_env is None:
                    return None
                api_key_env = api_key_env.strip()
        else:
            # No default, must specify
            api_key_env = questionary.text(
                "Enter API key environment variable name:",
                validate=lambda x: True if x.strip() else "Env var name is required",
            ).ask()
            if api_key_env is None:
                return None
            api_key_env = api_key_env.strip()

    return {
        "id": cred_id,
        "provider": actual_provider,
        "base_url": base_url,
        "api_key_env": api_key_env,
    }


# =============================================================================
# Commands - Order: add, test, list, fetch
# =============================================================================


@provider_app.command("add")
def add_credential(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="Provider type: openai, anthropic, gemini, ollama, openrouter",
        ),
    ] = None,
    cred_id: Annotated[
        str | None,
        typer.Option(
            "--id",
            help="Unique identifier for this credential",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            "-u",
            help="API base URL (required for OpenAI-compatible providers)",
        ),
    ] = None,
    api_key_env: Annotated[
        str | None,
        typer.Option(
            "--api-key-env",
            "-e",
            help="Environment variable name for API key",
        ),
    ] = None,
) -> None:
    """Add a new provider credential to markit.yaml.

    Run without arguments for interactive mode, or provide options directly.

    Examples:
        markit provider add
        markit provider add --provider openai --id my-openai
        markit provider add -p openai -u https://api.deepseek.com --id deepseek -e DEEPSEEK_API_KEY
    """
    from markit.cli.commands.model import add_model

    credential: dict[str, Any] | None = None

    # Check if running in interactive mode (no provider specified)
    if provider is None:
        credential = _interactive_add_credential()
        if credential is None:
            console.print("\n[yellow]Cancelled by user[/yellow]")
            raise typer.Exit()
    else:
        # Validate provider type
        valid_providers = ["openai", "anthropic", "gemini", "ollama", "openrouter"]
        if provider not in valid_providers:
            console.print(f"[red]Invalid provider type: {provider}[/red]")
            console.print(f"Valid types: {', '.join(valid_providers)}")
            raise typer.Exit(1)

        # Validate required fields
        if not cred_id:
            console.print("[red]--id is required when using command line arguments[/red]")
            raise typer.Exit(1)

        # Get provider defaults
        provider_info = PROVIDER_TYPES.get(provider, {})

        # Use defaults if not specified
        final_base_url = base_url or provider_info.get("base_url")
        final_api_key_env = api_key_env or provider_info.get("default_api_key_env")

        credential = {
            "id": cred_id,
            "provider": provider,
            "base_url": final_base_url,
            "api_key_env": final_api_key_env,
        }

    # Write to config
    if _write_credential_to_config(credential):
        console.print(
            f"\n[green]Successfully added credential '{credential['id']}' to markit.yaml![/green]"
        )

        # Ask if user wants to add a model now
        select_now = questionary.confirm(
            "Would you like to add a model for this credential now?",
            default=True,
        ).ask()

        if select_now:
            console.print()
            add_model()
        else:
            console.print("\nRun [bold]markit model add[/bold] later to add models.")
    else:
        raise typer.Exit(1)


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
    providers_to_test = get_unique_credentials(settings)

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
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Models", justify="right")
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

    Tests each provider configured in markit.yaml by calling their
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


@provider_app.command("list")
def list_credentials() -> None:
    """List all configured provider credentials."""
    settings = get_settings()
    creds = settings.llm.credentials

    if not creds:
        console.print("[yellow]No credentials configured in markit.yaml[/yellow]")
        console.print()
        console.print("Run [bold]markit provider add[/bold] to add a credential.")
        return

    table = Table(title="Configured Credentials")
    table.add_column("ID", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Base URL", style="dim")
    table.add_column("API Key Env")

    for cred in creds:
        table.add_row(
            cred.id,
            cred.provider,
            cred.base_url or "-",
            cred.api_key_env or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(creds)} credential(s)[/dim]")


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


@provider_app.command("fetch")
def fetch_models(
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
    """Fetch available models from all configured providers.

    Fetches the models list from each provider's API and saves
    to local cache (~/.cache/markit/models.json) by default.
    """
    settings = get_settings()
    all_models: dict[str, list[dict[str, Any]]] = {}

    async def fetch_all_models() -> None:
        creds = get_unique_credentials(settings)

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
