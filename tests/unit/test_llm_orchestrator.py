"""Tests for LLMOrchestrator service."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from markit.services.llm_orchestrator import LLMOrchestrator


class TestLLMOrchestratorInit:
    """Tests for LLMOrchestrator initialization."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM configuration."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return config

    def test_init_stores_config(self, mock_llm_config):
        """Initialization stores provided configuration."""
        orchestrator = LLMOrchestrator(llm_config=mock_llm_config)
        assert orchestrator.llm_config is mock_llm_config

    def test_init_with_cli_overrides(self, mock_llm_config):
        """CLI overrides are stored correctly."""
        orchestrator = LLMOrchestrator(
            llm_config=mock_llm_config,
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-5",
        )
        assert orchestrator.llm_provider == "anthropic"
        assert orchestrator.llm_model == "claude-sonnet-4-5"

    def test_init_lazy_components(self, mock_llm_config):
        """Components are not initialized until needed."""
        orchestrator = LLMOrchestrator(llm_config=mock_llm_config)
        assert orchestrator._provider_manager is None
        assert orchestrator._enhancer is None
        assert orchestrator._image_analyzer is None
        assert orchestrator._provider_manager_initialized is False


class TestGetRequiredCapabilities:
    """Tests for capability detection."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    def test_text_only(self, orchestrator):
        """Text enhancement only requires text capability."""
        caps = orchestrator._get_required_capabilities(llm_enabled=True, analyze_image=False)
        assert "text" in caps
        assert "vision" not in caps

    def test_vision_only(self, orchestrator):
        """Image analysis requires both text and vision."""
        caps = orchestrator._get_required_capabilities(llm_enabled=False, analyze_image=True)
        assert "text" in caps
        assert "vision" in caps

    def test_both_enabled(self, orchestrator):
        """Both features require text and vision."""
        caps = orchestrator._get_required_capabilities(llm_enabled=True, analyze_image=True)
        assert "text" in caps
        assert "vision" in caps

    def test_neither_enabled(self, orchestrator):
        """Default to text capability when nothing enabled."""
        caps = orchestrator._get_required_capabilities(llm_enabled=False, analyze_image=False)
        assert "text" in caps


class TestGetDefaultModel:
    """Tests for default model retrieval."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    def test_supported_provider(self, orchestrator):
        """Supported provider returns a default model."""
        # All these should not raise
        for provider in ["openai", "anthropic", "gemini", "ollama", "openrouter"]:
            model = orchestrator._get_default_model(provider)
            assert model is not None
            assert isinstance(model, str)
            assert len(model) > 0

    def test_unsupported_provider_raises(self, orchestrator):
        """Unsupported provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            orchestrator._get_default_model("unknown_provider")
        assert "unknown_provider" in str(exc_info.value)
        assert "Supported providers" in str(exc_info.value)


class TestWarmup:
    """Tests for warmup functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    async def test_warmup_skipped_when_disabled(self, orchestrator):
        """Warmup is skipped when LLM features are disabled."""
        with patch.object(orchestrator, "_get_provider_manager_sync") as mock_get_manager:
            await orchestrator.warmup(llm_enabled=False, analyze_image=False)
            mock_get_manager.assert_not_called()

    async def test_warmup_initializes_provider(self, orchestrator):
        """Warmup initializes provider manager with non-lazy mode."""
        mock_manager = AsyncMock()
        mock_manager.available_providers = ["gemini"]

        with patch.object(orchestrator, "_get_provider_manager_sync", return_value=mock_manager):
            await orchestrator.warmup(llm_enabled=True, analyze_image=False)

            mock_manager.initialize.assert_called_once()
            call_kwargs = mock_manager.initialize.call_args[1]
            assert call_kwargs["lazy"] is False
            assert "text" in call_kwargs["required_capabilities"]


class TestGetProviderManager:
    """Tests for provider manager retrieval."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    async def test_lazy_initialization(self, orchestrator):
        """Provider manager is lazily initialized."""
        mock_manager = AsyncMock()

        with patch.object(orchestrator, "_create_provider_manager", return_value=mock_manager):
            result = await orchestrator.get_provider_manager()

            assert result is mock_manager
            mock_manager.initialize.assert_called_once()

    async def test_cached_manager_reused(self, orchestrator):
        """Cached provider manager is reused on subsequent calls."""
        mock_manager = AsyncMock()

        with patch.object(orchestrator, "_create_provider_manager", return_value=mock_manager):
            result1 = await orchestrator.get_provider_manager()
            result2 = await orchestrator.get_provider_manager()

            assert result1 is result2
            # _create_provider_manager should only be called once
            orchestrator._create_provider_manager.assert_called_once()


class TestGetEnhancer:
    """Tests for enhancer retrieval."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    async def test_enhancer_created_lazily(self, orchestrator):
        """Enhancer is created lazily."""
        mock_manager = AsyncMock()

        with (
            patch.object(orchestrator, "get_provider_manager", return_value=mock_manager),
            patch("markit.llm.enhancer.MarkdownEnhancer") as MockEnhancer,
        ):
            # First call creates enhancer
            await orchestrator.get_enhancer()

            # Verify enhancer was created with provider manager
            MockEnhancer.assert_called_once()
            call_kwargs = MockEnhancer.call_args[1]
            assert call_kwargs["provider_manager"] is mock_manager


class TestGetImageAnalyzer:
    """Tests for image analyzer retrieval."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    async def test_analyzer_created_lazily(self, orchestrator):
        """Image analyzer is created lazily."""
        mock_manager = AsyncMock()

        with (
            patch.object(orchestrator, "get_provider_manager", return_value=mock_manager),
            patch("markit.image.analyzer.ImageAnalyzer") as MockAnalyzer,
        ):
            # First call creates analyzer
            await orchestrator.get_image_analyzer()

            # Verify analyzer was created with provider manager
            MockAnalyzer.assert_called_once()
            call_kwargs = MockAnalyzer.call_args[1]
            assert call_kwargs["provider_manager"] is mock_manager


class TestCreateLLMTasks:
    """Tests for LLM task creation."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    @pytest.fixture
    def mock_images(self):
        """Create mock compressed images."""
        return [
            Mock(filename="img1.png"),
            Mock(filename="img2.png"),
        ]

    async def test_creates_enhancement_task_when_enabled(self, orchestrator, mock_images):
        """Enhancement task is created when LLM is enabled."""
        mock_manager = AsyncMock()
        mock_manager.has_capability = Mock(return_value=True)

        with patch.object(orchestrator, "get_provider_manager", return_value=mock_manager):
            tasks = await orchestrator.create_llm_tasks(
                images_for_analysis=mock_images,
                markdown_content="# Test",
                input_file=Path("test.pdf"),
                llm_enabled=True,
                analyze_image=False,
            )

            # Should have 1 enhancement task
            assert len(tasks) == 1

            # Clean up unawaited coroutines
            for task in tasks:
                task.close()

    async def test_creates_analysis_tasks_when_enabled(self, orchestrator, mock_images):
        """Image analysis tasks are created when vision is enabled."""
        mock_manager = AsyncMock()
        mock_manager.has_capability = Mock(return_value=True)

        with patch.object(orchestrator, "get_provider_manager", return_value=mock_manager):
            tasks = await orchestrator.create_llm_tasks(
                images_for_analysis=mock_images,
                markdown_content="# Test",
                input_file=Path("test.pdf"),
                llm_enabled=False,
                analyze_image=True,
            )

            # Should have 2 analysis tasks (one per image)
            assert len(tasks) == 2

            # Clean up unawaited coroutines
            for task in tasks:
                task.close()

    async def test_creates_both_task_types(self, orchestrator, mock_images):
        """Both task types are created when both features are enabled."""
        mock_manager = AsyncMock()
        mock_manager.has_capability = Mock(return_value=True)

        with patch.object(orchestrator, "get_provider_manager", return_value=mock_manager):
            tasks = await orchestrator.create_llm_tasks(
                images_for_analysis=mock_images,
                markdown_content="# Test",
                input_file=Path("test.pdf"),
                llm_enabled=True,
                analyze_image=True,
            )

            # Should have 3 tasks: 2 analysis + 1 enhancement
            assert len(tasks) == 3

            # Clean up unawaited coroutines
            for task in tasks:
                task.close()

    async def test_skips_analysis_without_vision_capability(self, orchestrator, mock_images):
        """Image analysis is skipped when vision capability is unavailable."""
        mock_manager = AsyncMock()
        mock_manager.has_capability = Mock(return_value=False)

        with patch.object(orchestrator, "get_provider_manager", return_value=mock_manager):
            tasks = await orchestrator.create_llm_tasks(
                images_for_analysis=mock_images,
                markdown_content="# Test",
                input_file=Path("test.pdf"),
                llm_enabled=False,
                analyze_image=True,
            )

            # No tasks should be created
            assert len(tasks) == 0


class TestCreateEnhancementTask:
    """Tests for enhancement task execution."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    async def test_returns_enhanced_content(self, orchestrator):
        """Enhancement task returns enhanced content."""
        from markit.llm.enhancer import EnhancedMarkdown

        mock_enhancer = AsyncMock()
        mock_enhancer.enhance = AsyncMock(
            return_value=EnhancedMarkdown(content="# Enhanced", summary="Test doc")
        )

        with patch.object(orchestrator, "get_enhancer", return_value=mock_enhancer):
            result = await orchestrator.create_enhancement_task(
                markdown="# Original",
                source_file=Path("test.pdf"),
                return_stats=False,
            )

            assert result == "# Enhanced"

    async def test_returns_stats_when_requested(self, orchestrator):
        """Enhancement task returns stats when requested."""
        from markit.llm.base import LLMTaskResultWithStats
        from markit.llm.enhancer import EnhancedMarkdown

        mock_enhancer = AsyncMock()
        mock_enhancer.enhance = AsyncMock(
            return_value=LLMTaskResultWithStats(
                result=EnhancedMarkdown(content="# Enhanced", summary="Test"),
                model="test-model",
                prompt_tokens=100,
                completion_tokens=50,
            )
        )

        with patch.object(orchestrator, "get_enhancer", return_value=mock_enhancer):
            result = await orchestrator.create_enhancement_task(
                markdown="# Original",
                source_file=Path("test.pdf"),
                return_stats=True,
            )

            assert isinstance(result, LLMTaskResultWithStats)
            assert result.result == "# Enhanced"
            assert result.model == "test-model"

    async def test_fallback_on_error(self, orchestrator):
        """Enhancement falls back to simple cleaning on error."""
        mock_enhancer = AsyncMock()
        mock_enhancer.enhance = AsyncMock(side_effect=Exception("LLM Error"))

        with patch.object(orchestrator, "get_enhancer", return_value=mock_enhancer):
            result = await orchestrator.create_enhancement_task(
                markdown="# Original\n\n\n\nContent",
                source_file=Path("test.pdf"),
                return_stats=False,
            )

            # Should return cleaned content (not enhanced, but not failed)
            assert isinstance(result, str)
            assert len(result) > 0


class TestCreateImageAnalysisTask:
    """Tests for image analysis task execution."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    @pytest.fixture
    def mock_image(self):
        """Create a mock compressed image."""
        return Mock(filename="test.png", data=b"image_data")

    async def test_returns_analysis(self, orchestrator, mock_image):
        """Analysis task returns ImageAnalysis."""
        from markit.image.analyzer import ImageAnalysis

        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(
            return_value=ImageAnalysis(
                alt_text="Test image",
                detailed_description="A test image",
                detected_text=None,
                image_type="photo",
            )
        )

        with patch.object(orchestrator, "get_image_analyzer", return_value=mock_analyzer):
            result = await orchestrator.create_image_analysis_task(
                image=mock_image,
                return_stats=False,
            )

            assert isinstance(result, ImageAnalysis)
            assert result.alt_text == "Test image"

    async def test_fallback_on_error(self, orchestrator, mock_image):
        """Analysis falls back to basic info on error."""
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(side_effect=Exception("Analysis Error"))

        with patch.object(orchestrator, "get_image_analyzer", return_value=mock_analyzer):
            result = await orchestrator.create_image_analysis_task(
                image=mock_image,
                return_stats=False,
            )

            # Should return fallback analysis
            assert "test.png" in result.alt_text
            assert "failed" in result.detailed_description.lower()


class TestHasCapability:
    """Tests for capability checking."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock config."""
        config = Mock()
        config.providers = []
        config.model_copy = Mock(return_value=config)
        return LLMOrchestrator(llm_config=config)

    def test_no_manager_returns_false(self, orchestrator):
        """Returns False when provider manager not initialized."""
        assert orchestrator.has_capability("text") is False
        assert orchestrator.has_capability("vision") is False

    def test_delegates_to_manager(self, orchestrator):
        """Delegates capability check to provider manager."""
        mock_manager = Mock()
        mock_manager.has_capability = Mock(return_value=True)
        orchestrator._provider_manager = mock_manager

        result = orchestrator.has_capability("vision")

        assert result is True
        mock_manager.has_capability.assert_called_once_with("vision")
