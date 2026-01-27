"""Tests to verify config.schema.json is in sync with config.py models."""

import json
from pathlib import Path

import pytest

from markitai.config import (
    BatchConfig,
    CacheConfig,
    ImageConfig,
    LogConfig,
    OCRConfig,
    OutputConfig,
    PresetConfig,
    PromptsConfig,
    RouterSettings,
    ScreenshotConfig,
)

# Path to the schema file
SCHEMA_PATH = (
    Path(__file__).parent.parent.parent / "src" / "markitai" / "config.schema.json"
)


@pytest.fixture
def schema() -> dict:
    """Load the JSON schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


class TestSchemaSync:
    """Tests to verify schema is in sync with Pydantic models."""

    def test_schema_file_exists(self) -> None:
        """Verify schema file exists."""
        assert SCHEMA_PATH.exists(), f"Schema file not found: {SCHEMA_PATH}"

    def test_schema_is_valid_json(self, schema: dict) -> None:
        """Verify schema is valid JSON."""
        assert isinstance(schema, dict)
        assert "$defs" in schema
        assert "properties" in schema

    def test_image_config_alt_enabled_in_schema(self, schema: dict) -> None:
        """Verify ImageConfig.alt_enabled is in schema."""
        image_config = schema["$defs"]["ImageConfig"]["properties"]
        assert "alt_enabled" in image_config
        assert image_config["alt_enabled"]["default"] is False

    def test_image_config_desc_enabled_in_schema(self, schema: dict) -> None:
        """Verify ImageConfig.desc_enabled is in schema."""
        image_config = schema["$defs"]["ImageConfig"]["properties"]
        assert "desc_enabled" in image_config
        assert image_config["desc_enabled"]["default"] is False

    def test_screenshot_config_in_schema(self, schema: dict) -> None:
        """Verify ScreenshotConfig is in schema."""
        screenshot_config = schema["$defs"]["ScreenshotConfig"]["properties"]
        assert "enabled" in screenshot_config
        assert screenshot_config["enabled"]["default"] is False

    def test_prompts_config_system_user_pairs_in_schema(self, schema: dict) -> None:
        """Verify PromptsConfig has system/user prompt pairs in schema."""
        prompts_config = schema["$defs"]["PromptsConfig"]["properties"]
        # Check key system/user pairs exist
        assert "image_analysis_system" in prompts_config
        assert "image_analysis_user" in prompts_config
        assert "page_content_system" in prompts_config
        assert "page_content_user" in prompts_config
        assert "document_enhance_system" in prompts_config
        assert "document_enhance_user" in prompts_config
        assert "cleaner_system" in prompts_config
        assert "cleaner_user" in prompts_config

    def test_preset_config_definition_in_schema(self, schema: dict) -> None:
        """Verify PresetConfig is defined in schema."""
        assert "PresetConfig" in schema["$defs"]
        preset_config = schema["$defs"]["PresetConfig"]["properties"]
        assert "llm" in preset_config
        assert "ocr" in preset_config
        assert "alt" in preset_config
        assert "desc" in preset_config
        assert "screenshot" in preset_config

    def test_presets_property_in_root_schema(self, schema: dict) -> None:
        """Verify presets property is in root schema."""
        assert "presets" in schema["properties"]

    def test_log_config_default_level(self, schema: dict) -> None:
        """Verify LogConfig.level default matches config.py."""
        log_config = schema["$defs"]["LogConfig"]["properties"]
        # config.py has level = "INFO" (via DEFAULT_LOG_LEVEL constant)
        assert log_config["level"]["default"] == "INFO"

    def test_router_settings_defaults(self, schema: dict) -> None:
        """Verify RouterSettings defaults match config.py."""
        router_settings = schema["$defs"]["RouterSettings"]["properties"]
        # config.py has num_retries = 2, timeout = 120
        assert router_settings["num_retries"]["default"] == 2
        assert router_settings["timeout"]["default"] == 120


class TestModelFieldSync:
    """Tests to verify model fields exist in schema."""

    def test_image_config_fields_match(self, schema: dict) -> None:
        """Verify all ImageConfig fields are in schema."""
        model_fields = set(ImageConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["ImageConfig"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_ocr_config_fields_match(self, schema: dict) -> None:
        """Verify all OCRConfig fields are in schema."""
        model_fields = set(OCRConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["OCRConfig"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_screenshot_config_fields_match(self, schema: dict) -> None:
        """Verify all ScreenshotConfig fields are in schema."""
        model_fields = set(ScreenshotConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["ScreenshotConfig"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_prompts_config_fields_match(self, schema: dict) -> None:
        """Verify all PromptsConfig fields are in schema."""
        model_fields = set(PromptsConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["PromptsConfig"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_preset_config_fields_match(self, schema: dict) -> None:
        """Verify all PresetConfig fields are in schema."""
        model_fields = set(PresetConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["PresetConfig"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_log_config_fields_match(self, schema: dict) -> None:
        """Verify all LogConfig fields are in schema."""
        model_fields = set(LogConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["LogConfig"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_router_settings_fields_match(self, schema: dict) -> None:
        """Verify all RouterSettings fields are in schema."""
        model_fields = set(RouterSettings.model_fields.keys())
        schema_fields = set(schema["$defs"]["RouterSettings"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"

    def test_cache_config_fields_match(self, schema: dict) -> None:
        """Verify all CacheConfig fields are in schema."""
        model_fields = set(CacheConfig.model_fields.keys())
        schema_fields = set(schema["$defs"]["CacheConfig"]["properties"].keys())

        missing = model_fields - schema_fields
        assert not missing, f"Fields missing from schema: {missing}"


class TestSchemaDefaults:
    """Tests to verify schema defaults match Pydantic model defaults."""

    def test_output_config_defaults(self, schema: dict) -> None:
        """Verify OutputConfig defaults."""
        schema_props = schema["$defs"]["OutputConfig"]["properties"]
        model = OutputConfig()

        assert schema_props["dir"]["default"] == model.dir
        assert schema_props["on_conflict"]["default"] == model.on_conflict
        assert schema_props["allow_symlinks"]["default"] == model.allow_symlinks

    def test_batch_config_defaults(self, schema: dict) -> None:
        """Verify BatchConfig defaults."""
        schema_props = schema["$defs"]["BatchConfig"]["properties"]
        model = BatchConfig()

        assert schema_props["concurrency"]["default"] == model.concurrency
        assert (
            schema_props["state_flush_interval_seconds"]["default"]
            == model.state_flush_interval_seconds
        )
        assert schema_props["scan_max_depth"]["default"] == model.scan_max_depth
        assert schema_props["scan_max_files"]["default"] == model.scan_max_files

    def test_image_config_defaults(self, schema: dict) -> None:
        """Verify ImageConfig defaults."""
        schema_props = schema["$defs"]["ImageConfig"]["properties"]
        model = ImageConfig()

        assert schema_props["alt_enabled"]["default"] == model.alt_enabled
        assert schema_props["desc_enabled"]["default"] == model.desc_enabled
        assert schema_props["compress"]["default"] == model.compress
        assert schema_props["quality"]["default"] == model.quality
        assert schema_props["format"]["default"] == model.format

    def test_ocr_config_defaults(self, schema: dict) -> None:
        """Verify OCRConfig defaults."""
        schema_props = schema["$defs"]["OCRConfig"]["properties"]
        model = OCRConfig()

        assert schema_props["enabled"]["default"] == model.enabled
        assert schema_props["lang"]["default"] == model.lang

    def test_screenshot_config_defaults(self, schema: dict) -> None:
        """Verify ScreenshotConfig defaults."""
        schema_props = schema["$defs"]["ScreenshotConfig"]["properties"]
        model = ScreenshotConfig()

        assert schema_props["enabled"]["default"] == model.enabled

    def test_log_config_defaults(self, schema: dict) -> None:
        """Verify LogConfig defaults."""
        schema_props = schema["$defs"]["LogConfig"]["properties"]
        model = LogConfig()

        assert schema_props["level"]["default"] == model.level
        assert schema_props["dir"]["default"] == model.dir
        assert schema_props["rotation"]["default"] == model.rotation
        assert schema_props["retention"]["default"] == model.retention
