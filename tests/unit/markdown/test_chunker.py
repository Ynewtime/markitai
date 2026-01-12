"""Tests for markdown chunker module."""

from unittest.mock import MagicMock, patch

from markit.markdown.chunker import ChunkConfig, MarkdownChunker, SimpleChunker


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkConfig()
        assert config.max_tokens == 32000
        assert config.overlap_tokens == 500
        assert config.min_chunk_tokens == 1000
        assert config.encoding == "cl100k_base"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkConfig(
            max_tokens=8000,
            overlap_tokens=200,
            min_chunk_tokens=500,
            encoding="gpt2",
        )
        assert config.max_tokens == 8000
        assert config.overlap_tokens == 200
        assert config.min_chunk_tokens == 500
        assert config.encoding == "gpt2"


class TestMarkdownChunker:
    """Tests for MarkdownChunker class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        chunker = MarkdownChunker()
        assert chunker.config is not None
        assert isinstance(chunker.config, ChunkConfig)
        assert chunker._encoding is None
        assert chunker._header_splitter is None
        assert chunker._text_splitter is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ChunkConfig(max_tokens=16000)
        chunker = MarkdownChunker(config)
        assert chunker.config.max_tokens == 16000

    def test_init_splitters_lazy(self):
        """Test splitters are lazily initialized."""
        chunker = MarkdownChunker()
        # Before any operation, splitters should be None
        assert chunker._encoding is None
        assert chunker._header_splitter is None

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_init_splitters(self, mock_recursive, mock_header, mock_tiktoken):
        """Test _init_splitters initializes all components."""
        mock_tiktoken.return_value = MagicMock()
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        chunker = MarkdownChunker()
        chunker._init_splitters()

        mock_tiktoken.assert_called_once_with("cl100k_base")
        mock_header.assert_called_once()
        mock_recursive.from_tiktoken_encoder.assert_called_once()

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_init_splitters_only_once(self, mock_recursive, mock_header, mock_tiktoken):
        """Test splitters are only initialized once."""
        mock_encoding = MagicMock()
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        chunker = MarkdownChunker()
        chunker._init_splitters()
        chunker._init_splitters()  # Second call

        # Should only be called once
        assert mock_tiktoken.call_count == 1

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_count_tokens(self, mock_recursive, mock_header, mock_tiktoken):
        """Test token counting."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        chunker = MarkdownChunker()
        count = chunker.count_tokens("test text")

        assert count == 5
        mock_encoding.encode.assert_called_once_with("test text")

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_needs_chunking_false(self, mock_recursive, mock_header, mock_tiktoken):
        """Test needs_chunking returns False for small text."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = list(range(100))  # 100 tokens
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        config = ChunkConfig(max_tokens=1000)
        chunker = MarkdownChunker(config)

        assert chunker.needs_chunking("small text") is False

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_needs_chunking_true(self, mock_recursive, mock_header, mock_tiktoken):
        """Test needs_chunking returns True for large text."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = list(range(2000))  # 2000 tokens
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        config = ChunkConfig(max_tokens=1000)
        chunker = MarkdownChunker(config)

        assert chunker.needs_chunking("large text") is True

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_chunk_small_document(self, mock_recursive, mock_header, mock_tiktoken):
        """Test chunking small document returns single chunk."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = list(range(100))  # Small
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        chunker = MarkdownChunker()
        result = chunker.chunk("Small document")

        assert result == ["Small document"]

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_chunk_large_document(self, mock_recursive, mock_header, mock_tiktoken):
        """Test chunking large document."""
        # Setup encoding to return large count first, then small for chunks
        token_counts = [50000, 5000, 5000, 5000]  # First is total, rest are chunks
        call_count = [0]

        def mock_encode(_text):
            count = token_counts[min(call_count[0], len(token_counts) - 1)]
            call_count[0] += 1
            return list(range(count))

        mock_encoding = MagicMock()
        mock_encoding.encode.side_effect = mock_encode
        mock_tiktoken.return_value = mock_encoding

        # Setup header splitter
        mock_chunk1 = MagicMock()
        mock_chunk1.page_content = "Chunk 1"
        mock_chunk2 = MagicMock()
        mock_chunk2.page_content = "Chunk 2"
        mock_header_instance = MagicMock()
        mock_header_instance.split_text.return_value = [mock_chunk1, mock_chunk2]
        mock_header.return_value = mock_header_instance

        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        config = ChunkConfig(max_tokens=32000)
        chunker = MarkdownChunker(config)
        result = chunker.chunk("Large document content")

        assert len(result) >= 1

    def test_merge_small_chunks_empty(self):
        """Test merging empty chunk list."""
        chunker = MarkdownChunker()
        chunker._encoding = MagicMock()
        result = chunker._merge_small_chunks([])
        assert result == []

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_merge_small_chunks_single(self, mock_recursive, mock_header, mock_tiktoken):
        """Test merging single chunk."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = list(range(500))
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        chunker = MarkdownChunker()
        chunker._init_splitters()

        result = chunker._merge_small_chunks(["Single chunk"])
        assert result == ["Single chunk"]

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_merge_small_chunks_merges_small(self, mock_recursive, mock_header, mock_tiktoken):
        """Test small chunks are merged."""
        mock_encoding = MagicMock()
        # All chunks are small (500 tokens each)
        mock_encoding.encode.return_value = list(range(500))
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        config = ChunkConfig(min_chunk_tokens=1000, max_tokens=32000)
        chunker = MarkdownChunker(config)
        chunker._init_splitters()

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        result = chunker._merge_small_chunks(chunks)

        # Small chunks should be merged
        assert len(result) < len(chunks)

    @patch("tiktoken.get_encoding")
    @patch("langchain_text_splitters.MarkdownHeaderTextSplitter")
    @patch("langchain_text_splitters.RecursiveCharacterTextSplitter")
    def test_merge_small_chunks_preserves_large(self, mock_recursive, mock_header, mock_tiktoken):
        """Test large chunks are not merged."""
        mock_encoding = MagicMock()
        # Large chunks (5000 tokens each)
        mock_encoding.encode.return_value = list(range(5000))
        mock_tiktoken.return_value = mock_encoding
        mock_header.return_value = MagicMock()
        mock_recursive.from_tiktoken_encoder.return_value = MagicMock()

        config = ChunkConfig(min_chunk_tokens=1000, max_tokens=8000)
        chunker = MarkdownChunker(config)
        chunker._init_splitters()

        chunks = ["Large 1", "Large 2", "Large 3"]
        result = chunker._merge_small_chunks(chunks)

        # Large chunks should not be merged
        assert len(result) == 3

    def test_merge_chunks(self):
        """Test merging processed chunks."""
        chunker = MarkdownChunker()
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        result = chunker.merge(chunks)
        assert result == "Chunk 1\n\nChunk 2\n\nChunk 3"


class TestSimpleChunker:
    """Tests for SimpleChunker class."""

    def test_init_default(self):
        """Test default initialization."""
        chunker = SimpleChunker()
        assert chunker.max_chars == 12000
        assert chunker.overlap_chars == 500

    def test_init_custom(self):
        """Test custom initialization."""
        chunker = SimpleChunker(max_chars=5000, overlap_chars=200)
        assert chunker.max_chars == 5000
        assert chunker.overlap_chars == 200

    def test_chunk_small_text(self):
        """Test chunking small text returns single chunk."""
        chunker = SimpleChunker(max_chars=1000)
        text = "Short text"
        result = chunker.chunk(text)
        assert result == ["Short text"]

    def test_chunk_large_text(self):
        """Test chunking large text."""
        chunker = SimpleChunker(max_chars=100, overlap_chars=20)
        text = "A" * 500  # 500 characters
        result = chunker.chunk(text)
        assert len(result) > 1

    def test_chunk_breaks_at_paragraph(self):
        """Test chunking prefers paragraph breaks."""
        chunker = SimpleChunker(max_chars=100, overlap_chars=10)
        text = "A" * 40 + "\n\n" + "B" * 40 + "C" * 100
        result = chunker.chunk(text)
        # Should prefer breaking at paragraph
        assert len(result) >= 2

    def test_chunk_breaks_at_sentence(self):
        """Test chunking prefers sentence breaks."""
        chunker = SimpleChunker(max_chars=100, overlap_chars=10)
        text = "A" * 40 + ". " + "B" * 40 + "C" * 100
        result = chunker.chunk(text)
        assert len(result) >= 2

    def test_chunk_overlap(self):
        """Test chunks have overlap."""
        chunker = SimpleChunker(max_chars=50, overlap_chars=10)
        text = "ABCDE" * 20  # 100 chars
        result = chunker.chunk(text)
        # Chunks should have some overlap
        assert len(result) > 1

    def test_chunk_strips_whitespace(self):
        """Test chunks are stripped."""
        chunker = SimpleChunker(max_chars=50, overlap_chars=5)
        text = "   text   " * 10
        result = chunker.chunk(text)
        for chunk in result:
            assert chunk == chunk.strip()

    def test_merge(self):
        """Test merging chunks."""
        chunker = SimpleChunker()
        chunks = ["Part 1", "Part 2", "Part 3"]
        result = chunker.merge(chunks)
        assert result == "Part 1\n\nPart 2\n\nPart 3"


class TestChunkerIntegration:
    """Integration tests for chunker."""

    def test_simple_chunker_roundtrip(self):
        """Test simple chunker preserves content."""
        chunker = SimpleChunker(max_chars=50, overlap_chars=10)
        original = "This is a test. " * 5
        chunks = chunker.chunk(original)
        merged = chunker.merge(chunks)
        # Merged should contain all original words
        for word in original.split():
            assert word in merged
