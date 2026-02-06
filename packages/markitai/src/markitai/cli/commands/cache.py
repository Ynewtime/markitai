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

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.i18n import t
from markitai.config import ConfigManager
from markitai.constants import DEFAULT_CACHE_DB_FILENAME
from markitai.llm import SQLiteCache

console = get_console()


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def _print_verbose(cache_data: dict[str, Any], con: Console) -> None:
    """Print verbose cache details in compact format."""
    by_model = cache_data.get("by_model", {})
    entries = cache_data.get("entries", [])
    width = ui.term_width(con)

    if by_model:
        con.print()
        ui.section("By Model")
        max_name = min(max((len(m) for m in by_model), default=0), 20)
        for model, data in by_model.items():
            name_pad = ui.truncate(model, max_name).ljust(max_name)
            con.print(
                f"  [cyan]{name_pad}[/]  "
                f"{data['count']:>3} entries  "
                f"{format_size(data['size_bytes']):>8}"
            )

    if entries:
        con.print()
        ui.section(f"Recent Entries ({len(entries)})")
        # Fixed prefix: "  {key:8}  {model:10} {size:>8}" = ~34 chars
        prefix_len = 34
        preview_max = width - prefix_len
        show_preview = preview_max >= 15
        for entry in entries:
            key_short = entry["key"][:8]
            size = format_size(entry["size_bytes"])
            model = entry.get("model", "")
            line = f"  [dim]{key_short}[/]  {model:<10} {size:>8}"
            if show_preview:
                preview = entry.get("preview", "")
                if preview:
                    preview = preview.replace("\n", " ").strip()
                    line += f"  [dim]{ui.truncate(preview, preview_max)}[/]"
            con.print(line)


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
            if verbose:
                # Single query for stats + model breakdown
                stats_data["cache"] = global_cache.stats_verbose()
            else:
                stats_data["cache"] = global_cache.stats()
        except Exception as e:
            stats_data["cache"] = {"error": str(e)}

    # Collect verbose entries (separate query, different shape)
    if (
        verbose
        and global_cache
        and stats_data["cache"]
        and "error" not in stats_data["cache"]
    ):
        stats_data["cache"]["entries"] = global_cache.list_entries(limit)

    if as_json:
        click.echo(json.dumps(stats_data, indent=2, ensure_ascii=False))
    else:
        ui.title(t("cache.title"))

        console.print(f"  {t('enabled')}: {cfg.cache.enabled}")
        console.print()

        if stats_data["cache"]:
            c = stats_data["cache"]
            if "error" in c:
                ui.error(f"Cache: {c['error']}")
            else:
                ui.info(
                    f"{t('cache.llm')}: {c['count']} {t('cache.entries')} "
                    f"({c['size_mb']} MB)"
                )
        else:
            ui.info(f"{t('cache.llm')}: 0 {t('cache.entries')}")

        # Print verbose details after summary
        if verbose and stats_data.get("cache") and "error" not in stats_data["cache"]:
            _print_verbose(stats_data["cache"], console)


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
        ui.summary(t("cache.cleared", count=result["cache"]))
        if result["spa_domains"] > 0:
            console.print(f"  SPA domains: {result['spa_domains']}")
    else:
        console.print(f"[dim]{t('cache.no_entries')}[/dim]")


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
            click.echo(json.dumps({"cleared": count}))
        else:
            console.print(f"[green]Cleared {count} learned SPA domains[/green]")
        return

    domains = spa_cache.list_domains()

    if as_json:
        click.echo(json.dumps(domains, indent=2, ensure_ascii=False))
        return

    if not domains:
        console.print("[dim]No learned SPA domains yet[/dim]")
        console.print(
            "\n[dim]Domains are learned automatically when static fetch "
            "detects JavaScript requirement.[/dim]"
        )
        return

    ui.title(f"Learned SPA Domains ({len(domains)} {t('total').lower()})")

    for d in domains:
        status = "expired" if d.get("expired") else "active"
        if status == "active":
            ui.success(d["domain"])
        else:
            ui.warning(d["domain"], detail="Expired")

        learned_at = d.get("learned_at", "")[:10] if d.get("learned_at") else "-"
        hits = d.get("hits", 0)
        console.print(f"    [dim]│ Hits: {hits}, Learned: {learned_at}[/dim]")


__all__ = ["cache"]
