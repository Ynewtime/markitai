"""Service layer for markit.

This module provides service classes that encapsulate specific functionality
extracted from the ConversionPipeline, following the Single Responsibility Principle.

Services:
    - ImageProcessingService: Image format conversion, deduplication, compression
    - LLMOrchestrator: LLM provider management, enhancement, analysis coordination
    - OutputManager: File output, conflict resolution, asset management
"""

from markit.services.image_processor import ImageProcessingConfig, ImageProcessingService
from markit.services.llm_orchestrator import LLMOrchestrator
from markit.services.output_manager import OutputManager

__all__ = [
    "ImageProcessingConfig",
    "ImageProcessingService",
    "LLMOrchestrator",
    "OutputManager",
]
