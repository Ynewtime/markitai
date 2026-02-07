"""Tests for cache fingerprint collision resistance."""


class TestCacheFingerprint:
    """Test that cache fingerprints use robust hashing."""

    def test_different_documents_different_fingerprints(self):
        """Documents with same prefix but different content should have different fingerprints."""
        from markitai.llm.document import _compute_document_fingerprint

        # Two documents with identical first 1000 chars but different content
        prefix = "A" * 1000
        doc1 = prefix + " DOCUMENT ONE CONTENT"
        doc2 = prefix + " DOCUMENT TWO CONTENT"

        fp1 = _compute_document_fingerprint(doc1, ["page1"])
        fp2 = _compute_document_fingerprint(doc2, ["page1"])

        assert fp1 != fp2, (
            "Documents with same prefix should have different fingerprints"
        )

    def test_fingerprint_includes_page_info(self):
        """Fingerprint should incorporate page names."""
        from markitai.llm.document import _compute_document_fingerprint

        doc = "Same content"
        fp1 = _compute_document_fingerprint(doc, ["page1", "page2"])
        fp2 = _compute_document_fingerprint(doc, ["page1", "page2", "page3"])

        assert fp1 != fp2, "Different page counts should produce different fingerprints"

    def test_fingerprint_uses_sha256(self):
        """Fingerprint should use SHA256 for collision resistance."""
        from markitai.llm.document import _compute_document_fingerprint

        fp = _compute_document_fingerprint("test content", ["page1"])
        # SHA256 hex digest is 64 characters
        assert len(fp) == 64

    def test_fingerprint_deterministic(self):
        """Same input should always produce same fingerprint."""
        from markitai.llm.document import _compute_document_fingerprint

        fp1 = _compute_document_fingerprint("hello world", ["p1", "p2"])
        fp2 = _compute_document_fingerprint("hello world", ["p1", "p2"])
        assert fp1 == fp2

    def test_empty_content(self):
        """Empty content should produce valid fingerprint."""
        from markitai.llm.document import _compute_document_fingerprint

        fp = _compute_document_fingerprint("", [])
        assert len(fp) == 64
