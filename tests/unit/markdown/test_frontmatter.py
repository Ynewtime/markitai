"""Tests for frontmatter module."""

from datetime import datetime
from pathlib import Path

from markit.markdown.frontmatter import (
    FRONTMATTER_PATTERN,
    Frontmatter,
    FrontmatterHandler,
    ImageDescriptionFrontmatter,
    KnowledgeGraphData,
    _escape_yaml_string,
    create_frontmatter,
    inject_frontmatter,
)


class TestFrontmatterPattern:
    """Tests for FRONTMATTER_PATTERN regex."""

    def test_matches_basic_frontmatter(self):
        """Test matching basic frontmatter."""
        text = "---\ntitle: Test\n---\n\nContent"
        match = FRONTMATTER_PATTERN.match(text)
        assert match is not None
        assert "title: Test" in match.group(1)

    def test_no_match_without_frontmatter(self):
        """Test no match when frontmatter is absent."""
        text = "# Title\n\nContent"
        match = FRONTMATTER_PATTERN.match(text)
        assert match is None

    def test_no_match_mid_document(self):
        """Test frontmatter must be at start."""
        text = "Content\n---\ntitle: Test\n---\n"
        match = FRONTMATTER_PATTERN.match(text)
        assert match is None


class TestFrontmatter:
    """Tests for Frontmatter dataclass."""

    def test_default_values(self):
        """Test default values."""
        fm = Frontmatter()
        assert fm.title is None
        assert fm.processed is None
        assert fm.description is None
        assert fm.source is None
        assert fm.entities is None
        assert fm.topics is None
        assert fm.domain is None
        assert fm.extra == {}

    def test_custom_values(self):
        """Test custom values."""
        fm = Frontmatter(
            title="Test Title",
            processed="2025-01-01",
            description="A description",
            source="test.pdf",
            entities=["Entity1", "Entity2"],
            topics=["Topic1"],
            domain="tech",
            extra={"custom_field": "value"},
        )
        assert fm.title == "Test Title"
        assert fm.entities == ["Entity1", "Entity2"]
        assert fm.extra["custom_field"] == "value"

    def test_to_dict_basic(self):
        """Test to_dict with basic fields."""
        fm = Frontmatter(title="Test", processed="2025-01-01")
        d = fm.to_dict()
        assert d["title"] == "Test"
        assert d["processed"] == "2025-01-01"
        assert "description" not in d  # None fields excluded

    def test_to_dict_with_extra(self):
        """Test to_dict includes extra fields."""
        fm = Frontmatter(title="Test", extra={"custom": "value"})
        d = fm.to_dict()
        assert d["title"] == "Test"
        assert d["custom"] == "value"

    def test_to_dict_knowledge_graph_fields(self):
        """Test to_dict includes knowledge graph fields."""
        fm = Frontmatter(
            entities=["Person", "Place"],
            topics=["History"],
            domain="humanities",
        )
        d = fm.to_dict()
        assert d["entities"] == ["Person", "Place"]
        assert d["topics"] == ["History"]
        assert d["domain"] == "humanities"

    def test_from_dict_basic(self):
        """Test from_dict with basic fields."""
        data = {"title": "Test", "processed": "2025-01-01"}
        fm = Frontmatter.from_dict(data)
        assert fm.title == "Test"
        assert fm.processed == "2025-01-01"

    def test_from_dict_with_extra(self):
        """Test from_dict extracts extra fields."""
        data = {"title": "Test", "custom_field": "custom_value"}
        fm = Frontmatter.from_dict(data)
        assert fm.title == "Test"
        assert fm.extra["custom_field"] == "custom_value"

    def test_from_dict_knowledge_graph(self):
        """Test from_dict with knowledge graph fields."""
        data = {
            "entities": ["E1"],
            "topics": ["T1"],
            "domain": "D1",
        }
        fm = Frontmatter.from_dict(data)
        assert fm.entities == ["E1"]
        assert fm.topics == ["T1"]
        assert fm.domain == "D1"

    def test_to_yaml(self):
        """Test YAML conversion."""
        fm = Frontmatter(title="Test Title", processed="2025-01-01")
        yaml_str = fm.to_yaml()
        assert "title: Test Title" in yaml_str
        assert "processed: '2025-01-01'" in yaml_str or "processed: 2025-01-01" in yaml_str


class TestFrontmatterHandler:
    """Tests for FrontmatterHandler class."""

    def test_init(self):
        """Test handler initialization."""
        handler = FrontmatterHandler()
        assert handler is not None

    def test_has_frontmatter_true(self):
        """Test detecting existing frontmatter."""
        handler = FrontmatterHandler()
        text = "---\ntitle: Test\n---\n\nContent"
        assert handler.has_frontmatter(text) is True

    def test_has_frontmatter_false(self):
        """Test detecting missing frontmatter."""
        handler = FrontmatterHandler()
        text = "# Title\n\nContent"
        assert handler.has_frontmatter(text) is False

    def test_parse_with_frontmatter(self):
        """Test parsing document with frontmatter."""
        handler = FrontmatterHandler()
        text = "---\ntitle: Test\ndescription: Desc\n---\n\nContent"
        fm, content = handler.parse(text)
        assert fm is not None
        assert fm.title == "Test"
        assert fm.description == "Desc"
        assert content.strip() == "Content"

    def test_parse_without_frontmatter(self):
        """Test parsing document without frontmatter."""
        handler = FrontmatterHandler()
        text = "# Title\n\nContent"
        fm, content = handler.parse(text)
        assert fm is None
        assert content == text

    def test_parse_empty_frontmatter(self):
        """Test parsing empty frontmatter."""
        handler = FrontmatterHandler()
        text = "---\n\n---\n\nContent"
        fm, content = handler.parse(text)
        assert fm is not None
        assert fm.to_dict() == {}

    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML frontmatter."""
        handler = FrontmatterHandler()
        text = "---\ninvalid: yaml: content:\n---\n\nContent"
        fm, content = handler.parse(text)
        # Should return None and original text on parse error
        assert fm is None

    def test_extract_with_frontmatter(self):
        """Test extracting frontmatter as dict."""
        handler = FrontmatterHandler()
        text = "---\ntitle: Test\n---\n\nContent"
        data = handler.extract(text)
        assert data["title"] == "Test"

    def test_extract_without_frontmatter(self):
        """Test extracting returns empty dict when no frontmatter."""
        handler = FrontmatterHandler()
        text = "# Title"
        data = handler.extract(text)
        assert data == {}

    def test_remove_frontmatter(self):
        """Test removing frontmatter."""
        handler = FrontmatterHandler()
        text = "---\ntitle: Test\n---\n\nContent"
        content = handler.remove(text)
        assert content == "Content"
        assert "---" not in content

    def test_remove_no_frontmatter(self):
        """Test remove when no frontmatter exists."""
        handler = FrontmatterHandler()
        text = "Content"
        content = handler.remove(text)
        assert content == "Content"

    def test_add_frontmatter(self):
        """Test adding frontmatter."""
        handler = FrontmatterHandler()
        fm = Frontmatter(title="New Title")
        result = handler.add("Content", fm)
        assert result.startswith("---")
        assert "title: New Title" in result
        assert "Content" in result

    def test_add_frontmatter_from_dict(self):
        """Test adding frontmatter from dict."""
        handler = FrontmatterHandler()
        result = handler.add("Content", {"title": "Dict Title"})
        assert "title: Dict Title" in result

    def test_add_frontmatter_replace_existing(self):
        """Test replacing existing frontmatter."""
        handler = FrontmatterHandler()
        text = "---\ntitle: Old\n---\n\nContent"
        fm = Frontmatter(title="New")
        result = handler.add(text, fm, replace=True)
        assert "title: New" in result
        assert "title: Old" not in result

    def test_add_frontmatter_no_replace(self):
        """Test adding without replacing existing."""
        handler = FrontmatterHandler()
        text = "---\ntitle: Old\n---\n\nContent"
        fm = Frontmatter(title="New")
        result = handler.add(text, fm, replace=False)
        # When not replacing, existing frontmatter content is kept
        assert "Content" in result

    def test_update_existing(self):
        """Test updating existing frontmatter."""
        handler = FrontmatterHandler()
        text = "---\ntitle: Original\n---\n\nContent"
        result = handler.update(text, {"description": "Added"})
        assert "title: Original" in result
        assert "description: Added" in result

    def test_update_create_new(self):
        """Test update creates frontmatter if none exists."""
        handler = FrontmatterHandler()
        text = "Content"
        result = handler.update(text, {"title": "New"})
        assert "---" in result
        assert "title: New" in result


class TestCreateFrontmatter:
    """Tests for create_frontmatter function."""

    def test_basic_creation(self):
        """Test basic frontmatter creation."""
        source = Path("/path/to/document.pdf")
        fm = create_frontmatter(source)
        assert fm.title == "document"
        assert fm.source == "document.pdf"
        assert fm.processed is not None

    def test_with_summary(self):
        """Test creation with summary."""
        source = Path("/path/to/doc.pdf")
        fm = create_frontmatter(source, summary="A summary")
        assert fm.description == "A summary"

    def test_with_extra(self):
        """Test creation with extra fields."""
        source = Path("/path/to/doc.pdf")
        fm = create_frontmatter(source, extra={"custom": "value"})
        assert fm.extra["custom"] == "value"

    def test_processed_date_format(self):
        """Test processed date is in correct format."""
        source = Path("/path/to/doc.pdf")
        fm = create_frontmatter(source)
        # Should be YYYY-MM-DD format
        assert fm.processed is not None
        assert len(fm.processed) == 10
        assert fm.processed[4] == "-"
        assert fm.processed[7] == "-"


class TestInjectFrontmatter:
    """Tests for inject_frontmatter function."""

    def test_inject_basic(self):
        """Test basic injection."""
        result = inject_frontmatter("Content", Path("test.pdf"))
        assert "---" in result
        assert "title: test" in result
        assert "source: test.pdf" in result
        assert "Content" in result

    def test_inject_with_summary(self):
        """Test injection with summary."""
        result = inject_frontmatter("Content", Path("test.pdf"), "A summary")
        assert "description:" in result


class TestEscapeYamlString:
    """Tests for _escape_yaml_string function."""

    def test_escape_none(self):
        """Test None input."""
        assert _escape_yaml_string(None) is None

    def test_escape_backslash(self):
        """Test backslash escaping."""
        result = _escape_yaml_string("path\\to\\file")
        assert result is not None and "\\\\" in result

    def test_escape_quotes(self):
        """Test quote escaping."""
        result = _escape_yaml_string('say "hello"')
        assert result is not None and '\\"' in result

    def test_escape_newlines(self):
        """Test newline replacement."""
        result = _escape_yaml_string("line1\nline2")
        assert result is not None
        assert "\n" not in result
        assert " " in result

    def test_escape_carriage_return(self):
        """Test carriage return removal."""
        result = _escape_yaml_string("line1\rline2")
        assert result is not None and "\r" not in result


class TestKnowledgeGraphData:
    """Tests for KnowledgeGraphData dataclass."""

    def test_default_values(self):
        """Test default values."""
        kg = KnowledgeGraphData()
        assert kg.entities is None
        assert kg.relationships is None
        assert kg.topics is None
        assert kg.domain is None

    def test_to_dict_empty(self):
        """Test to_dict with no values."""
        kg = KnowledgeGraphData()
        assert kg.to_dict() == {}

    def test_to_dict_with_values(self):
        """Test to_dict with values."""
        kg = KnowledgeGraphData(
            entities=["E1", "E2"],
            relationships=["R1"],
            topics=["T1"],
            domain="D1",
        )
        d = kg.to_dict()
        assert d["entities"] == ["E1", "E2"]
        assert d["relationships"] == ["R1"]
        assert d["topics"] == ["T1"]
        assert d["domain"] == "D1"

    def test_to_dict_partial(self):
        """Test to_dict with partial values."""
        kg = KnowledgeGraphData(entities=["E1"])
        d = kg.to_dict()
        assert d == {"entities": ["E1"]}


class TestImageDescriptionFrontmatter:
    """Tests for ImageDescriptionFrontmatter class."""

    def test_create_basic(self):
        """Test basic creation."""
        handler = ImageDescriptionFrontmatter()
        result = handler.create(
            image_filename="test.png",
            image_type="diagram",
            alt_text="A diagram",
            detailed_description="Detailed description here",
        )
        assert "---" in result
        assert "source_image: test.png" in result
        assert "image_type: diagram" in result
        assert "# Image Description" in result
        assert "## Short Description" in result
        assert "A diagram" in result
        assert "## Detailed Description" in result
        assert "Detailed description here" in result

    def test_create_with_detected_text(self):
        """Test creation with detected text."""
        handler = ImageDescriptionFrontmatter()
        result = handler.create(
            image_filename="test.png",
            image_type="photo",
            alt_text="A photo",
            detailed_description="Description",
            detected_text="OCR text here",
        )
        assert "## Detected Text" in result
        assert "OCR text here" in result

    def test_create_with_knowledge_graph(self):
        """Test creation with knowledge graph data."""
        handler = ImageDescriptionFrontmatter()
        kg = KnowledgeGraphData(
            entities=["Entity1"],
            relationships=["Entity1 -> Entity2"],
            topics=["Topic1"],
            domain="tech",
        )
        result = handler.create(
            image_filename="test.png",
            image_type="diagram",
            alt_text="Alt",
            detailed_description="Desc",
            knowledge_graph=kg,
        )
        # Frontmatter should contain KG metadata
        assert "entities:" in result
        assert "topics:" in result
        assert "domain: tech" in result
        # Should have relationships section
        assert "## Relationships" in result
        assert "Entity1 -> Entity2" in result

    def test_create_kg_without_relationships(self):
        """Test KG metadata without relationships section."""
        handler = ImageDescriptionFrontmatter()
        kg = KnowledgeGraphData(
            entities=["Entity1"],
            topics=["Topic1"],
        )
        result = handler.create(
            image_filename="test.png",
            image_type="diagram",
            alt_text="Alt",
            detailed_description="Desc",
            knowledge_graph=kg,
        )
        assert "entities:" in result
        assert "## Relationships" not in result

    def test_create_generated_at_format(self):
        """Test generated_at date format."""
        handler = ImageDescriptionFrontmatter()
        result = handler.create(
            image_filename="test.png",
            image_type="photo",
            alt_text="Alt",
            detailed_description="Desc",
        )
        assert "generated_at:" in result
        # Check date format YYYY-MM-DD
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in result
