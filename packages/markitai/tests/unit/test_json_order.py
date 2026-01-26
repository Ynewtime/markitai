"""Tests for JSON ordering module."""

from markitai.json_order import (
    _format_duration,
    _order_llm_usage,
    _transform_file_entry,
    _transform_summary,
    _transform_url_entry,
    order_dict,
    order_dict_keys_sorted,
    order_images,
    order_report,
    order_state,
)


class TestFormatDuration:
    """Tests for _format_duration function."""

    def test_none_returns_zero(self) -> None:
        """Test that None returns '0s'."""
        assert _format_duration(None) == "0s"

    def test_short_duration(self) -> None:
        """Test short durations under 60 seconds."""
        assert _format_duration(0) == "0.0s"
        assert _format_duration(1.5) == "1.5s"
        assert _format_duration(32.7) == "32.7s"
        assert _format_duration(59.9) == "59.9s"

    def test_minutes_format(self) -> None:
        """Test durations between 1 and 60 minutes."""
        assert _format_duration(60) == "01:00"
        assert _format_duration(90) == "01:30"
        assert _format_duration(186) == "03:06"
        assert _format_duration(3599) == "59:59"

    def test_hours_format(self) -> None:
        """Test durations over 1 hour."""
        assert _format_duration(3600) == "01:00:00"
        assert _format_duration(3661) == "01:01:01"
        assert _format_duration(7325) == "02:02:05"


class TestOrderDict:
    """Tests for order_dict function."""

    def test_basic_ordering(self) -> None:
        """Test basic field ordering."""
        d = {"c": 3, "a": 1, "b": 2}
        order = ["a", "b", "c"]
        result = order_dict(d, order)
        assert list(result.keys()) == ["a", "b", "c"]

    def test_partial_ordering(self) -> None:
        """Test ordering with fields not in order list."""
        d = {"c": 3, "a": 1, "extra": 4, "b": 2}
        order = ["a", "b", "c"]
        result = order_dict(d, order)
        # Ordered fields first, then extras
        keys = list(result.keys())
        assert keys[:3] == ["a", "b", "c"]
        assert "extra" in keys

    def test_non_dict_passthrough(self) -> None:
        """Test that non-dict values pass through unchanged."""
        assert order_dict("string", ["a"]) == "string"  # type: ignore
        assert order_dict(123, ["a"]) == 123  # type: ignore


class TestOrderDictKeysSorted:
    """Tests for order_dict_keys_sorted function."""

    def test_alphabetical_sort(self) -> None:
        """Test alphabetical key sorting."""
        d = {"zebra": 1, "alpha": 2, "beta": 3}
        result = order_dict_keys_sorted(d)
        assert list(result.keys()) == ["alpha", "beta", "zebra"]


class TestTransformSummary:
    """Tests for _transform_summary function."""

    def test_duration_formatting(self) -> None:
        """Test duration fields are converted to human-readable format."""
        summary = {
            "total_documents": 10,
            "completed_documents": 8,
            "failed_documents": 2,
            "duration": 186.5,
            "processing_time": 450.0,
        }
        result = _transform_summary(summary)

        assert result["duration"] == "03:06"
        assert result["processing_time"] == "07:30"

    def test_preserves_other_fields(self) -> None:
        """Test that other fields are preserved."""
        summary = {
            "total_documents": 10,
            "completed_documents": 8,
            "failed_documents": 2,
            "duration": 60,
        }
        result = _transform_summary(summary)
        assert result["total_documents"] == 10
        assert result["completed_documents"] == 8
        assert result["failed_documents"] == 2


class TestOrderLlmUsage:
    """Tests for _order_llm_usage function."""

    def test_orders_fields(self) -> None:
        """Test LLM usage field ordering."""
        llm_usage = {
            "cost_usd": 0.05,
            "requests": 10,
            "models": {},
            "input_tokens": 1000,
            "output_tokens": 500,
        }
        result = _order_llm_usage(llm_usage)

        keys = list(result.keys())
        assert keys[0] == "models"
        assert "requests" in keys
        assert "cost_usd" in keys

    def test_orders_nested_models(self) -> None:
        """Test that nested model usage is ordered."""
        llm_usage = {
            "models": {
                "gpt-4": {"cost_usd": 0.05, "requests": 10, "input_tokens": 1000}
            },
            "requests": 10,
        }
        result = _order_llm_usage(llm_usage)

        model_keys = list(result["models"]["gpt-4"].keys())
        assert model_keys[0] == "requests"


class TestTransformFileEntry:
    """Tests for _transform_file_entry function."""

    def test_duration_formatting(self) -> None:
        """Test file entry duration formatting."""
        entry = {
            "status": "completed",
            "output": "output.md",
            "duration": 45.2,
            "images": 5,
            "cost_usd": 0.01,
        }
        result = _transform_file_entry(entry)

        assert result["duration"] == "45.2s"
        assert result["images"] == 5
        assert result["cost_usd"] == 0.01

    def test_orders_fields(self) -> None:
        """Test that fields are ordered correctly."""
        entry = {
            "duration": 10.0,
            "status": "completed",
            "output": "test.md",
        }
        result = _transform_file_entry(entry)

        keys = list(result.keys())
        assert keys[0] == "status"


class TestTransformUrlEntry:
    """Tests for _transform_url_entry function."""

    def test_cache_details_merging(self) -> None:
        """Test that fetch_cache_hit and llm_cache_hit are merged into cache_details."""
        entry = {
            "status": "completed",
            "fetch_cache_hit": True,
            "llm_cache_hit": False,
            "duration": 3.5,
            "fetch_strategy": "browser",
        }
        result = _transform_url_entry(entry)

        assert result["cache_hit"] is True
        assert result["cache_details"] == {"fetch": True, "llm": False}
        assert result["fetch_strategy"] == "browser"
        assert result["duration"] == "3.5s"
        assert "fetch_cache_hit" not in result
        assert "llm_cache_hit" not in result

    def test_no_cache_fields(self) -> None:
        """Test URL entry without cache fields."""
        entry = {
            "status": "completed",
            "duration": 10.0,
        }
        result = _transform_url_entry(entry)

        assert result["duration"] == "10.0s"
        # cache_details not added if no cache fields present
        assert "cache_details" not in result


class TestOrderReport:
    """Tests for order_report function."""

    def test_files_renamed_to_documents(self) -> None:
        """Test that 'files' is renamed to 'documents' for non-URL files."""
        report = {
            "version": "1.0",
            "documents": {
                "/path/to/file.pdf": {
                    "status": "completed",
                    "duration": 10.0,
                }
            },
        }
        result = order_report(report)

        assert "documents" in result
        assert "/path/to/file.pdf" in result["documents"]

    def test_urls_converted_to_url_sources_hierarchy(self) -> None:
        """Test that flat 'urls' dict is converted to hierarchical 'url_sources'."""
        report = {
            "version": "1.0",
            "urls": {
                "https://example.com/a": {
                    "source_file": "sources.urls",
                    "status": "completed",
                    "duration": 5.0,
                },
                "https://example.com/b": {
                    "source_file": "sources.urls",
                    "status": "failed",
                    "error": "Timeout",
                    "duration": 30.0,
                },
                "https://other.com/c": {
                    "source_file": "other.urls",
                    "status": "completed",
                    "duration": 2.0,
                },
            },
        }
        result = order_report(report)

        # urls should be converted to url_sources
        assert "urls" not in result

        # url_sources should be hierarchical
        assert "url_sources" in result
        assert "sources.urls" in result["url_sources"]
        assert "other.urls" in result["url_sources"]

        # Check sources.urls entry
        sources_entry = result["url_sources"]["sources.urls"]
        assert sources_entry["total"] == 2
        assert sources_entry["completed"] == 1
        assert sources_entry["failed"] == 1
        assert "https://example.com/a" in sources_entry["urls"]
        assert "https://example.com/b" in sources_entry["urls"]

        # Check other.urls entry
        other_entry = result["url_sources"]["other.urls"]
        assert other_entry["total"] == 1
        assert other_entry["completed"] == 1
        assert other_entry["failed"] == 0

        # Check that source_file is removed from URL entries
        assert "source_file" not in sources_entry["urls"]["https://example.com/a"]

    def test_full_report_transformation(self) -> None:
        """Test full report transformation with all fields."""
        report = {
            "version": "1.0",
            "started_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:05:00",
            "options": {
                "llm": True,
                "cache": True,
                "concurrency": 4,
            },
            "summary": {
                "total_documents": 5,
                "completed_documents": 4,
                "failed_documents": 1,
                "duration": 300.0,
                "processing_time": 450.0,
            },
            "llm_usage": {
                "requests": 20,
                "input_tokens": 5000,
                "output_tokens": 2000,
                "cost_usd": 0.10,
                "models": {},
            },
            "documents": {
                "doc.pdf": {
                    "status": "completed",
                    "duration": 60.0,
                }
            },
        }
        result = order_report(report)

        # Check top-level field order
        keys = list(result.keys())
        assert keys[0] == "version"
        assert "options" in keys
        assert "summary" in keys
        assert "llm_usage" in keys

        # Check summary transformation (duration formatting)
        assert result["summary"]["duration"] == "05:00"
        assert result["summary"]["processing_time"] == "07:30"


class TestOrderState:
    """Tests for order_state function."""

    def test_preserves_numeric_duration(self) -> None:
        """Test that state.json preserves numeric duration values."""
        state = {
            "version": "1.0",
            "documents": {
                "/path/file.pdf": {
                    "status": "completed",
                    "duration": 10.5,
                    "images": 3,
                }
            },
            "urls": {
                "https://example.com": {
                    "status": "completed",
                    "source_file": "test.urls",
                    "duration": 5.0,
                }
            },
        }
        result = order_state(state)

        # Should have 'documents'
        assert "documents" in result

        # Should keep 'urls' flat, not hierarchical
        assert "urls" in result

        # Duration should stay numeric (not formatted)
        assert result["documents"]["/path/file.pdf"]["duration"] == 10.5
        assert result["urls"]["https://example.com"]["duration"] == 5.0


class TestOrderImages:
    """Tests for order_images function."""

    def test_images_ordering(self) -> None:
        """Test images.json ordering."""
        images = {
            "version": "1.0",
            "created": "2024-01-01T00:00:00",
            "updated": "2024-01-01T00:05:00",
            "images": [
                {
                    "source": "doc.pdf",
                    "path": "image1.png",
                    "alt": "An image",
                    "desc": "Description",
                    "llm_usage": {
                        "gpt-4": {
                            "requests": 1,
                            "input_tokens": 100,
                            "output_tokens": 50,
                        }
                    },
                }
            ],
        }
        result = order_images(images)

        # Check top-level order
        keys = list(result.keys())
        assert keys[0] == "version"
        assert "images" in keys

        # Check image entry ordering
        image = result["images"][0]
        image_keys = list(image.keys())
        assert image_keys[0] == "path"
