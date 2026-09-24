"""Tests for URL list parsing edge cases."""

from __future__ import annotations

from pathlib import Path

from markitai.urls import parse_url_list

_BOM = "﻿"


class TestUrlListBom:
    """Windows editors prepend a UTF-8 BOM; it must not cost the first URL."""

    def test_bom_text_list_keeps_first_url(self, tmp_path: Path) -> None:
        url_file = tmp_path / "list.urls"
        url_file.write_text(
            f"{_BOM}https://a.example/\nhttps://b.example/ named\n", encoding="utf-8"
        )

        entries = parse_url_list(url_file)

        assert [entry.url for entry in entries] == [
            "https://a.example/",
            "https://b.example/",
        ]
        assert entries[1].output_name == "named"

    def test_bom_json_array_is_parsed_as_json(self, tmp_path: Path) -> None:
        url_file = tmp_path / "list.urls"
        url_file.write_text(
            f'{_BOM}["https://a.example/", {{"url": "https://b.example/"}}]',
            encoding="utf-8",
        )

        entries = parse_url_list(url_file)

        assert [entry.url for entry in entries] == [
            "https://a.example/",
            "https://b.example/",
        ]
