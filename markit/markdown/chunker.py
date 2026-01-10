"""Large document chunking using LangChain Text Splitters."""

from dataclasses import dataclass

from markit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    max_tokens: int = 32000  # Optimized for large context models like Gemini 3 Flash
    overlap_tokens: int = 500
    min_chunk_tokens: int = 1000  # Minimum tokens before merging with adjacent chunk
    encoding: str = "cl100k_base"  # tiktoken encoding (cl100k_base for GPT-4 class models)


class MarkdownChunker:
    """Chunk large Markdown documents for processing.

    Uses LangChain Text Splitters for intelligent chunking that:
    - Respects Markdown structure (headers, code blocks, etc.)
    - Maintains semantic coherence
    - Handles token counting accurately
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """Initialize the chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        self._encoding = None
        self._header_splitter = None
        self._text_splitter = None

    def _init_splitters(self) -> None:
        """Lazy initialization of splitters."""
        if self._header_splitter is not None:
            return

        import tiktoken
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        # Get tokenizer using encoding name directly
        self._encoding = tiktoken.get_encoding(self.config.encoding)

        # Markdown header splitter
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )

        # Recursive text splitter for long sections
        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.config.encoding,
            chunk_size=self.config.max_tokens,
            chunk_overlap=self.config.overlap_tokens,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        self._init_splitters()
        return len(self._encoding.encode(text))

    def needs_chunking(self, text: str) -> bool:
        """Check if text needs to be chunked.

        Args:
            text: Text to check

        Returns:
            True if text exceeds max tokens
        """
        return self.count_tokens(text) > self.config.max_tokens

    def chunk(self, markdown: str) -> list[str]:
        """Chunk a Markdown document.

        Strategy:
        1. First split by Markdown headers to preserve structure
        2. For any chunks that are still too long, use recursive splitting
        3. Merge small adjacent chunks to reduce total chunk count

        Args:
            markdown: Markdown content to chunk

        Returns:
            List of chunks
        """
        self._init_splitters()

        # If document is small enough, return as single chunk
        if not self.needs_chunking(markdown):
            log.debug("Document small enough, no chunking needed")
            return [markdown]

        total_tokens = self.count_tokens(markdown)
        log.info(
            "Chunking document",
            total_tokens=total_tokens,
            max_tokens=self.config.max_tokens,
        )

        # First, split by headers
        header_chunks = self._header_splitter.split_text(markdown)

        # Process each chunk - split long ones
        intermediate_chunks = []
        for chunk in header_chunks:
            content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            token_count = self.count_tokens(content)

            if token_count > self.config.max_tokens:
                # Chunk is too long, split further
                log.debug(
                    "Section too long, splitting further",
                    tokens=token_count,
                )
                sub_chunks = self._text_splitter.split_text(content)
                intermediate_chunks.extend(sub_chunks)
            else:
                intermediate_chunks.append(content)

        # Merge small adjacent chunks to reduce LLM calls
        final_chunks = self._merge_small_chunks(intermediate_chunks)

        log.info(
            "Document chunked",
            num_chunks=len(final_chunks),
            before_merge=len(intermediate_chunks),
        )

        return final_chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge small adjacent chunks to reduce total chunk count.

        This prevents excessive LLM calls when documents have many small sections.

        Args:
            chunks: List of chunks to potentially merge

        Returns:
            List of merged chunks
        """
        if not chunks:
            return chunks

        min_tokens = self.config.min_chunk_tokens
        max_tokens = self.config.max_tokens
        merged = []
        current = chunks[0]
        current_tokens = self.count_tokens(current)

        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            next_tokens = self.count_tokens(next_chunk)

            # Try to merge if current chunk is small and combined size is within limit
            combined_tokens = current_tokens + next_tokens
            if current_tokens < min_tokens and combined_tokens <= max_tokens:
                # Merge chunks
                current = current + "\n\n" + next_chunk
                current_tokens = combined_tokens
            else:
                # Save current and start new
                merged.append(current)
                current = next_chunk
                current_tokens = next_tokens

        # Don't forget the last chunk
        merged.append(current)

        if len(merged) < len(chunks):
            log.debug(
                "Merged small chunks",
                original=len(chunks),
                merged=len(merged),
            )

        return merged

    def merge(self, chunks: list[str]) -> str:
        """Merge processed chunks back together.

        Args:
            chunks: List of processed chunks

        Returns:
            Merged content
        """
        return "\n\n".join(chunks)


class SimpleChunker:
    """Simple character-based chunker as fallback."""

    def __init__(self, max_chars: int = 12000, overlap_chars: int = 500) -> None:
        """Initialize simple chunker.

        Args:
            max_chars: Maximum characters per chunk
            overlap_chars: Overlap between chunks
        """
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> list[str]:
        """Chunk text by character count.

        Args:
            text: Text to chunk

        Returns:
            List of chunks
        """
        if len(text) <= self.max_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chars

            # Try to break at a paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.max_chars // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break > start + self.max_chars // 2:
                        end = sentence_break + 2

            chunks.append(text[start:end].strip())
            start = end - self.overlap_chars

        return chunks

    def merge(self, chunks: list[str]) -> str:
        """Merge chunks back together."""
        return "\n\n".join(chunks)
