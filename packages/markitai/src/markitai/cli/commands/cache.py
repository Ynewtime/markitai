"""Cache management CLI commands.

This module provides CLI commands for managing Markitai cache:
- cache stats: Show cache statistics
- cache clear: Clear cache entries
- cache spa-domains: View or manage learned SPA domains
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from markitai.config import ConfigManager
from markitai.constants import DEFAULT_CACHE_DB_FILENAME
from markitai.llm import SQLiteCache

console = Console()


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


@click.group()
def cache() -> None:
    """Cache management commands."""
    pass


@cache.command("stats")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed cache entries and model breakdown.",
)
@click.option(
    "--limit",
    default=20,
    type=int,
    help="Number of entries to show in verbose mode (default: 20).",
)
def cache_stats(as_json: bool, verbose: bool, limit: int) -> None:
    """Show cache statistics."""

    def print_verbose_details(
        cache_obj: SQLiteCache, cache_name: str, limit: int, as_json: bool
    ) -> dict[str, Any]:
        """Collect and optionally print verbose cache details."""
        by_model = cache_obj.stats_by_model()
        entries = cache_obj.list_entries(limit)

        if not as_json:
            # Print By Model table
            if by_model:
                model_table = Table(title=f"{cache_name} - By Model")
                model_table.add_column("Model", style="cyan")
                model_table.add_column("Entries", justify="right")
                model_table.add_column("Size", justify="right")
                for model, data in by_model.items():
                    model_table.add_row(
                        model, str(data["count"]), format_size(data["size_bytes"])
                    )
                console.print(model_table)
                console.print()

            # Print Recent Entries table
            if entries:
                entry_table = Table(title=f"{cache_name} - Recent Entries")
                entry_table.add_column("Key", style="dim", max_width=18)
                entry_table.add_column("Model", max_width=30)
                entry_table.add_column("Size", justify="right")
                entry_table.add_column("Preview", max_width=40)
                for entry in entries:
                    key_display = (
                        entry["key"][:16] + "..."
                        if len(entry["key"]) > 16
                        else entry["key"]
                    )
                    entry_table.add_row(
                        key_display,
                        entry["model"],
                        format_size(entry["size_bytes"]),
                        entry["preview"],
                    )
                console.print(entry_table)

        return {"by_model": by_model, "entries": entries}

    manager = ConfigManager()
    cfg = manager.load()

    stats_data: dict[str, Any] = {
        "cache": None,
        "enabled": cfg.cache.enabled,
    }

    # Check global cache
    global_cache: SQLiteCache | None = None
    global_cache_path = (
        Path(cfg.cache.global_dir).expanduser() / DEFAULT_CACHE_DB_FILENAME
    )
    if global_cache_path.exists():
        try:
            global_cache = SQLiteCache(global_cache_path, cfg.cache.max_size_bytes)
            stats_data["cache"] = global_cache.stats()
        except Exception as e:
            stats_data["cache"] = {"error": str(e)}

    # Collect verbose data if needed
    if (
        verbose
        and global_cache
        and stats_data["cache"]
        and "error" not in stats_data["cache"]
    ):
        verbose_data = print_verbose_details(global_cache, "Cache", limit, as_json)
        stats_data["cache"]["by_model"] = verbose_data["by_model"]
        stats_data["cache"]["entries"] = verbose_data["entries"]

    if as_json:
        # Use soft_wrap=True to prevent rich from breaking long lines
        console.print(
            json.dumps(stats_data, indent=2, ensure_ascii=False), soft_wrap=True
        )
    else:
        console.print("[bold]Cache Statistics[/bold]")
        console.print(f"Enabled: {cfg.cache.enabled}")
        console.print()

        if stats_data["cache"]:
            c = stats_data["cache"]
            if "error" in c:
                console.print(f"[red]Cache error:[/red] {c['error']}")
            else:
                console.print(f"  Path: {c['db_path']}")
                console.print(f"  Entries: {c['count']}")
                console.print(f"  Size: {c['size_mb']} MB / {c['max_size_mb']} MB")
        else:
            console.print("[dim]No cache found[/dim]")


@cache.command("clear")
@click.option(
    "--include-spa-domains",
    is_flag=True,
    help="Also clear learned SPA domains.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def cache_clear(include_spa_domains: bool, yes: bool) -> None:
    """Clear cache entries."""
    manager = ConfigManager()
    cfg = manager.load()

    # Confirm if not --yes
    if not yes:
        desc = "global cache (~/.markitai)"
        if include_spa_domains:
            desc += " + learned SPA domains"
        if not click.confirm(f"Clear {desc}?"):
            console.print("[yellow]Aborted[/yellow]")
            return

    result = {"cache": 0, "spa_domains": 0}

    # Clear global cache
    global_cache_path = (
        Path(cfg.cache.global_dir).expanduser() / DEFAULT_CACHE_DB_FILENAME
    )
    if global_cache_path.exists():
        try:
            global_cache = SQLiteCache(global_cache_path, cfg.cache.max_size_bytes)
            result["cache"] = global_cache.clear()
        except Exception as e:
            console.print(f"[red]Failed to clear cache:[/red] {e}")

    # Clear SPA domains if requested
    if include_spa_domains:
        from markitai.fetch import get_spa_domain_cache

        try:
            spa_cache = get_spa_domain_cache()
            result["spa_domains"] = spa_cache.clear()
        except Exception as e:
            console.print(f"[red]Failed to clear SPA domains:[/red] {e}")

    # Report results
    if result["cache"] > 0 or result["spa_domains"] > 0:
        console.print(f"[green]Cleared {result['cache']} cache entries[/green]")
        if result["spa_domains"] > 0:
            console.print(f"  SPA domains: {result['spa_domains']}")
    else:
        console.print("[dim]No cache entries to clear[/dim]")


@cache.command("spa-domains")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON.",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Clear all learned SPA domains.",
)
def cache_spa_domains(as_json: bool, clear: bool) -> None:
    """View or manage learned SPA domains.

    Shows domains that were automatically detected as requiring browser
    rendering (JavaScript-heavy sites). These domains will use browser
    strategy directly on future requests, avoiding wasted static fetch attempts.
    """
    from markitai.fetch import get_spa_domain_cache

    spa_cache = get_spa_domain_cache()

    if clear:
        count = spa_cache.clear()
        if as_json:
            console.print(json.dumps({"cleared": count}))
        else:
            console.print(f"[green]Cleared {count} learned SPA domains[/green]")
        return

    domains = spa_cache.list_domains()

    if as_json:
        console.print(json.dumps(domains, indent=2, ensure_ascii=False), soft_wrap=True)
        return

    if not domains:
        console.print("[dim]No learned SPA domains yet[/dim]")
        console.print(
            "\n[dim]Domains are learned automatically when static fetch "
            "detects JavaScript requirement.[/dim]"
        )
        return

    console.print(f"[bold]Learned SPA Domains[/bold] ({len(domains)} total)\n")

    table = Table()
    table.add_column("Domain", style="cyan")
    table.add_column("Hits", justify="right")
    table.add_column("Learned At", style="dim")
    table.add_column("Last Hit", style="dim")
    table.add_column("Status")

    for d in domains:
        status = "[red]Expired[/red]" if d.get("expired") else "[green]Active[/green]"
        learned_at = d.get("learned_at", "")[:10] if d.get("learned_at") else "-"
        last_hit = d.get("last_hit", "")[:10] if d.get("last_hit") else "-"
        table.add_row(
            d["domain"],
            str(d.get("hits", 0)),
            learned_at,
            last_hit,
            status,
        )

    console.print(table)
    console.print(
        "\n[dim]Tip: Use --clear to reset learned domains, "
        "or configure fallback_patterns in config file for permanent rules.[/dim]"
    )


__all__ = ["cache"]
