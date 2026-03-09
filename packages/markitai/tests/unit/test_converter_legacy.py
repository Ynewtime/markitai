from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from markitai.converter.base import ConvertResult, ExtractedImage
from markitai.converter.legacy import PptConverter


class TestLegacyPptArtifacts:
    """Tests for legacy PPT artifact naming."""

    def test_convert_rewrites_pptx_artifacts_to_original_ppt_name(
        self, tmp_path: Path
    ) -> None:
        """Legacy PPT outputs should not leak temporary .pptx filenames."""
        output_dir = tmp_path / "output"
        assets_dir = output_dir / ".markitai" / "assets"
        screenshots_dir = output_dir / ".markitai" / "screenshots"
        assets_dir.mkdir(parents=True)
        screenshots_dir.mkdir(parents=True)

        old_asset = assets_dir / "sample.pptx.0001.jpg"
        old_asset.write_bytes(b"asset")
        old_screenshot = screenshots_dir / "sample.pptx.slide0001.jpg"
        old_screenshot.write_bytes(b"screenshot")

        result = ConvertResult(
            markdown=(
                "![Embedded](.markitai/assets/sample.pptx.0001.jpg)\n"
                "<!-- ![Page 1](.markitai/screenshots/sample.pptx.slide0001.jpg) -->"
            ),
            images=[
                ExtractedImage(
                    path=old_asset,
                    index=1,
                    original_name=old_asset.name,
                    mime_type="image/jpeg",
                    width=10,
                    height=10,
                ),
                ExtractedImage(
                    path=old_screenshot,
                    index=1,
                    original_name=old_screenshot.name,
                    mime_type="image/jpeg",
                    width=10,
                    height=10,
                ),
            ],
            metadata={
                "page_images": [
                    {
                        "page": 1,
                        "path": str(old_screenshot),
                        "name": old_screenshot.name,
                    }
                ]
            },
        )

        converter = PptConverter()
        converter._pptx_converter = MagicMock()
        converter._pptx_converter.convert.return_value = result

        input_path = tmp_path / "sample.ppt"
        input_path.write_bytes(b"ppt")
        converted_path = tmp_path / "sample.pptx"

        with patch.object(
            converter,
            "_convert_legacy_format",
            return_value=converted_path,
        ):
            converted = converter.convert(input_path, output_dir)

        new_asset = assets_dir / "sample.ppt.0001.jpg"
        new_screenshot = screenshots_dir / "sample.ppt.slide0001.jpg"

        assert ".markitai/assets/sample.ppt.0001.jpg" in converted.markdown
        assert ".markitai/screenshots/sample.ppt.slide0001.jpg" in converted.markdown
        assert new_asset.exists()
        assert new_screenshot.exists()
        assert not old_asset.exists()
        assert not old_screenshot.exists()
        assert converted.images[0].path == new_asset
        assert converted.images[0].original_name == new_asset.name
        assert converted.images[1].path == new_screenshot
        assert converted.metadata["page_images"][0]["name"] == new_screenshot.name
        assert converted.metadata["page_images"][0]["path"] == str(new_screenshot)
