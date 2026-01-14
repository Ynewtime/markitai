"""Model management commands for listing and adding models."""

from pathlib import Path
from typing import Any

import questionary
import typer
from rich.console import Console
from rich.table import Table

from markit.cli.shared import get_unique_credentials
from markit.config import get_settings
from markit.config.constants import (
    DEFAULT_HTTP_CLIENT_TIMEOUT,
    DEFAULT_MODEL_LIST_TIMEOUT,
)
from markit.utils.capabilities import infer_capabilities
from markit.utils.logging import get_logger

model_app = typer.Typer(
    name="model",
    help="Manage LLM models.",
    no_args_is_help=True,
)

console = Console()
log = get_logger(__name__)


def _get_models_sync(
    provider_name: str, api_key: str | None, base_url: str | None
) -> list[dict[str, Any]]:
    """Get models list using synchronous API calls to avoid event loop issues."""
    models = []

    if provider_name == "openai" and api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=DEFAULT_MODEL_LIST_TIMEOUT)
        result = client.models.list()
        models = [{"id": m.id, "created": m.created, "owned_by": m.owned_by} for m in result.data]

    elif provider_name == "anthropic" and api_key:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key, timeout=DEFAULT_MODEL_LIST_TIMEOUT)
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
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=DEFAULT_MODEL_LIST_TIMEOUT,
        )
        result = client.models.list()
        models = [{"id": m.id, "created": m.created, "owned_by": m.owned_by} for m in result.data]

    elif provider_name == "ollama":
        import httpx

        host = base_url or "http://localhost:11434"
        with httpx.Client(timeout=DEFAULT_HTTP_CLIENT_TIMEOUT) as client:
            response = client.get(f"{host}/api/tags")
            data = response.json()
            models = [
                {"id": m.get("name"), "size": m.get("size"), "modified_at": m.get("modified_at")}
                for m in data.get("models", [])
            ]

    return models


@model_app.command("add")
def add_model() -> None:
    """Interactive wizard to select and configure a model.

    Allows you to:
    1. Select an existing provider/credential
    2. Browse available models (fetched live)
    3. Add the selected model to your markit.yaml
    """
    settings = get_settings()
    creds = get_unique_credentials(settings)

    if not creds:
        console.print("[red]No credentials configured![/red]")
        console.print()
        console.print(
            "To use this command, you need to configure at least one credential in markit.yaml."
        )
        console.print()
        console.print(
            "[bold]Quick fix:[/bold] Run [cyan]markit provider add[/cyan] to add a credential."
        )
        console.print()
        console.print("Or add a credential manually to your markit.yaml under the llm section:")
        console.print()
        console.print("[dim]llm:[/dim]")
        console.print("[dim]  credentials:[/dim]")
        console.print('[dim]    - id: "deepseek"[/dim]')
        console.print('[dim]      provider: "openai"[/dim]')
        console.print('[dim]      base_url: "https://api.deepseek.com"[/dim]')
        console.print('[dim]      api_key_env: "DEEPSEEK_API_KEY"[/dim]')
        console.print()
        console.print("See [cyan]markit.example.yaml[/cyan] for more credential examples.")
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
    model_display_name = questionary.text(
        "Enter a display name for this configuration:",
        default=default_name,
    ).ask()

    if model_display_name is None:
        # User cancelled with Ctrl+C
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit()

    # 5. Write to Config using ruamel.yaml to preserve comments and structure
    config_path = Path("markit.yaml")
    if not config_path.exists():
        console.print("[red]markit.yaml not found![/red]")
        raise typer.Exit(1)

    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedSeq

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Load the configuration (explicit UTF-8 for Windows compatibility)
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.load(f)

        # Prepare the new model entry
        new_model = {
            "name": model_display_name or real_model_id,
            "model": real_model_id,
            "credential_id": credential_id,
        }

        # Always explicitly declare capabilities for clarity
        new_model["capabilities"] = infer_capabilities(real_model_id)

        # Ensure llm structure exists
        if "llm" not in config_data:
            config_data["llm"] = {}

        llm_section = config_data["llm"]

        # Handle case where models key doesn't exist
        # We need to insert it after credentials to maintain proper YAML structure
        if "models" not in llm_section:
            # Create a new CommentedSeq for models
            models_seq = CommentedSeq()

            # Find the position after credentials to insert models
            # ruamel.yaml preserves key order, so we rebuild the section
            if "credentials" in llm_section:
                # Insert models key right after credentials
                new_llm = type(llm_section)()
                for key in llm_section:
                    new_llm[key] = llm_section[key]
                    if key == "credentials":
                        new_llm["models"] = models_seq
                # If credentials wasn't found (shouldn't happen), just add at end
                if "models" not in new_llm:
                    new_llm["models"] = models_seq
                config_data["llm"] = new_llm
                llm_section = config_data["llm"]
            else:
                llm_section["models"] = models_seq

        # Add the new model
        # Check if model already exists to avoid duplicates
        exists = False
        for m in llm_section["models"]:
            if m.get("name") == new_model["name"] and m.get("model") == new_model["model"]:
                exists = True
                break

        if not exists:
            llm_section["models"].append(new_model)
            # Write with explicit UTF-8 for Windows compatibility
            with config_path.open("w", encoding="utf-8") as f:
                yaml.dump(config_data, f)
            console.print(f"\n[green]Successfully added {real_model_id} to markit.yaml![/green]")
            console.print(
                "Run [bold]markit provider test[/bold] to verify the provider configuration,"
            )
            console.print("or verify the new model entry in markit.yaml directly.")
        else:
            console.print(
                f"\n[yellow]Model {real_model_id} already exists in configuration.[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Failed to update config: {e}[/red]")
        raise typer.Exit(1) from None


@model_app.command("list")
def list_models() -> None:
    """List all configured models in markit.yaml."""
    settings = get_settings()
    models = settings.llm.models

    if not models:
        console.print("[yellow]No models configured in markit.yaml[/yellow]")
        console.print()
        console.print("Run [bold]markit model add[/bold] to add a model.")
        return

    table = Table(title="Configured Models")
    table.add_column("Name", style="cyan")
    table.add_column("Model ID", style="dim")
    table.add_column("Credential", style="green")
    table.add_column("Capabilities")
    table.add_column("Timeout", justify="right")

    for model in models:
        caps = ", ".join(model.capabilities) if model.capabilities else "text"
        table.add_row(
            model.name,
            model.model,
            model.credential_id,
            caps,
            f"{model.timeout}s",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} model(s)[/dim]")
