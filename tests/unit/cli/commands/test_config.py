"""Tests for config command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from markit.cli.commands.config import (
    DEFAULT_CONFIG_TEMPLATE,
    config_app,
)


class TestConfigInit:
    """Tests for config init command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_init_creates_config(self, tmp_path):
        """Test that init creates a config file."""
        runner = CliRunner()
        config_path = tmp_path / "markit.yaml"

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(config_app, ["init", "--path", str(config_path)])
            assert result.exit_code == 0
            assert config_path.exists()
            assert "Created config file" in result.stdout

    def test_init_default_path(self, tmp_path):
        """Test init with default path."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(config_app, ["init"])
            assert result.exit_code == 0
            assert Path("markit.yaml").exists()

    def test_init_exists_no_force(self, tmp_path):
        """Test init fails if file exists without force."""
        runner = CliRunner()
        config_path = tmp_path / "markit.yaml"
        config_path.write_text("existing: config")

        result = runner.invoke(config_app, ["init", "--path", str(config_path)])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

    def test_init_force_overwrite(self, tmp_path):
        """Test init with force overwrites existing file."""
        runner = CliRunner()
        config_path = tmp_path / "markit.yaml"
        config_path.write_text("existing: config")

        result = runner.invoke(config_app, ["init", "--path", str(config_path), "--force"])
        assert result.exit_code == 0
        # Should contain new content
        content = config_path.read_text()
        assert "MarkIt Configuration" in content

    def test_init_template_content(self, tmp_path):
        """Test that init creates valid config template."""
        runner = CliRunner()
        config_path = tmp_path / "markit.yaml"

        result = runner.invoke(config_app, ["init", "--path", str(config_path)])
        assert result.exit_code == 0

        content = config_path.read_text()
        assert "log_level" in content
        assert "image:" in content
        assert "pdf:" in content
        assert "output:" in content


class TestConfigTest:
    """Tests for config test command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_test_valid_config(self, runner):
        """Test config test with valid configuration."""
        mock_settings = MagicMock()
        mock_settings.llm.credentials = []
        mock_settings.llm.models = []
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["test"])
            assert result.exit_code == 0
            assert "Configuration is valid" in result.stdout

    def test_test_with_credentials(self, runner):
        """Test config test with credentials configured."""
        mock_cred = MagicMock()
        mock_cred.id = "test-cred"
        mock_cred.provider = "openai"

        mock_settings = MagicMock()
        mock_settings.llm.credentials = [mock_cred]
        mock_settings.llm.models = []
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["test"])
            assert result.exit_code == 0
            assert "test-cred" in result.stdout

    def test_test_with_models(self, runner):
        """Test config test with models configured."""
        mock_model = MagicMock()
        mock_model.name = "gpt-4o"
        mock_model.model = "gpt-4o"
        mock_model.capabilities = ["text", "vision"]

        mock_settings = MagicMock()
        mock_settings.llm.credentials = []
        mock_settings.llm.models = [mock_model]
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["test"])
            assert result.exit_code == 0
            assert "gpt-4o" in result.stdout

    def test_test_with_legacy_providers(self, runner):
        """Test config test with legacy providers."""
        mock_provider = MagicMock()
        mock_provider.provider = "openai"
        mock_provider.api_key = "test-key"

        mock_settings = MagicMock()
        mock_settings.llm.credentials = []
        mock_settings.llm.models = []
        mock_settings.llm.providers = [mock_provider]

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["test"])
            assert result.exit_code == 0
            assert "Legacy" in result.stdout

    def test_test_no_providers(self, runner):
        """Test config test with no providers configured."""
        mock_settings = MagicMock()
        mock_settings.llm.credentials = []
        mock_settings.llm.models = []
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["test"])
            assert result.exit_code == 0
            assert "No LLM providers configured" in result.stdout

    def test_test_config_error(self, runner):
        """Test config test with configuration error."""
        with patch(
            "markit.cli.commands.config.get_settings", side_effect=Exception("Invalid config")
        ):
            result = runner.invoke(config_app, ["test"])
            assert result.exit_code == 1
            assert "Configuration error" in result.stdout


class TestConfigList:
    """Tests for config list command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_list_config(self, runner):
        """Test listing configuration."""
        mock_settings = MagicMock()
        mock_settings.log_level = "INFO"
        mock_settings.log_dir = ".logs"
        mock_settings.state_file = ".markit-state.json"
        mock_settings.output.default_dir = "output"
        mock_settings.output.on_conflict = "rename"
        mock_settings.output.create_assets_subdir = True
        mock_settings.image.enable_compression = True
        mock_settings.image.enable_analysis = False
        mock_settings.image.max_dimension = 2048
        mock_settings.pdf.engine = "pymupdf4llm"
        mock_settings.pdf.extract_images = True
        mock_settings.enhancement.enabled = False
        mock_settings.enhancement.add_frontmatter = True
        mock_settings.prompt.output_language = "zh"
        mock_settings.prompt.prompts_dir = "prompts"
        mock_settings.concurrency.file_workers = 4
        mock_settings.concurrency.image_workers = 8
        mock_settings.concurrency.llm_workers = 5
        mock_settings.llm.credentials = []
        mock_settings.llm.models = []
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["list"])
            assert result.exit_code == 0
            assert "Current Configuration" in result.stdout
            assert "INFO" in result.stdout
            assert "pymupdf4llm" in result.stdout

    def test_list_with_credentials(self, runner):
        """Test listing configuration with credentials."""
        mock_cred = MagicMock()
        mock_cred.id = "my-openai"
        mock_cred.provider = "openai"

        mock_settings = MagicMock()
        mock_settings.log_level = "INFO"
        mock_settings.log_dir = ".logs"
        mock_settings.state_file = ".markit-state.json"
        mock_settings.output.default_dir = "output"
        mock_settings.output.on_conflict = "rename"
        mock_settings.output.create_assets_subdir = True
        mock_settings.image.enable_compression = True
        mock_settings.image.enable_analysis = False
        mock_settings.image.max_dimension = 2048
        mock_settings.pdf.engine = "pymupdf4llm"
        mock_settings.pdf.extract_images = True
        mock_settings.enhancement.enabled = False
        mock_settings.enhancement.add_frontmatter = True
        mock_settings.prompt.output_language = "zh"
        mock_settings.prompt.prompts_dir = "prompts"
        mock_settings.concurrency.file_workers = 4
        mock_settings.concurrency.image_workers = 8
        mock_settings.concurrency.llm_workers = 5
        mock_settings.llm.credentials = [mock_cred]
        mock_settings.llm.models = []
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["list"])
            assert result.exit_code == 0
            assert "my-openai" in result.stdout


class TestConfigLocations:
    """Tests for config locations command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_locations_output(self, runner):
        """Test locations command output."""
        result = runner.invoke(config_app, ["locations"])
        assert result.exit_code == 0
        assert "Configuration File Locations" in result.stdout
        assert "markit.yaml" in result.stdout


class TestConfigAliases:
    """Tests for deprecated command aliases."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_show_alias(self, runner):
        """Test 'show' alias for 'list'."""
        mock_settings = MagicMock()
        mock_settings.log_level = "INFO"
        mock_settings.log_dir = ".logs"
        mock_settings.state_file = ".markit-state.json"
        mock_settings.output.default_dir = "output"
        mock_settings.output.on_conflict = "rename"
        mock_settings.output.create_assets_subdir = True
        mock_settings.image.enable_compression = True
        mock_settings.image.enable_analysis = False
        mock_settings.image.max_dimension = 2048
        mock_settings.pdf.engine = "pymupdf4llm"
        mock_settings.pdf.extract_images = True
        mock_settings.enhancement.enabled = False
        mock_settings.enhancement.add_frontmatter = True
        mock_settings.prompt.output_language = "zh"
        mock_settings.prompt.prompts_dir = "prompts"
        mock_settings.concurrency.file_workers = 4
        mock_settings.concurrency.image_workers = 8
        mock_settings.concurrency.llm_workers = 5
        mock_settings.llm.credentials = []
        mock_settings.llm.models = []
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["show"])
            assert result.exit_code == 0
            assert "deprecated" in result.stdout

    def test_validate_alias(self, runner):
        """Test 'validate' alias for 'test'."""
        mock_settings = MagicMock()
        mock_settings.llm.credentials = []
        mock_settings.llm.models = []
        mock_settings.llm.providers = []

        with patch("markit.cli.commands.config.get_settings", return_value=mock_settings):
            result = runner.invoke(config_app, ["validate"])
            assert result.exit_code == 0
            assert "deprecated" in result.stdout


class TestDefaultConfigTemplate:
    """Tests for DEFAULT_CONFIG_TEMPLATE constant."""

    def test_template_is_valid_yaml(self):
        """Test that the template is valid YAML."""
        from ruamel.yaml import YAML

        yaml = YAML()
        # Should not raise
        data = yaml.load(DEFAULT_CONFIG_TEMPLATE)
        assert "log_level" in data
        assert "image" in data
        assert "pdf" in data

    def test_template_has_required_sections(self):
        """Test that template has all required sections."""
        assert "log_level" in DEFAULT_CONFIG_TEMPLATE
        assert "image:" in DEFAULT_CONFIG_TEMPLATE
        assert "pdf:" in DEFAULT_CONFIG_TEMPLATE
        assert "output:" in DEFAULT_CONFIG_TEMPLATE
        assert "enhancement:" in DEFAULT_CONFIG_TEMPLATE
        assert "concurrency:" in DEFAULT_CONFIG_TEMPLATE
        assert "prompt:" in DEFAULT_CONFIG_TEMPLATE
