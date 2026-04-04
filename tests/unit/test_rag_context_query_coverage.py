"""Coverage tests for RAG context_builder and query_enhancer modules.

Targets uncovered paths in:
- alpha_zero/rag/agents/context_builder.py (52% → ~90%)
- alpha_zero/rag/agents/query_enhancer.py (57% → ~90%)
"""

from __future__ import annotations

from alpha_zero.rag.agents.context_builder import (
    ContextBuilder,
    ContextStrategy,
    ContextWindow,
)
from alpha_zero.rag.agents.query_enhancer import (
    QueryEnhancer,
    QueryExpansion,
)
from alpha_zero.rag.stores.base import Document, DocumentMetadata, SearchResult

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_result(content: str, score: float, source: str = "default") -> SearchResult:
    doc = Document(content=content)
    doc.metadata = DocumentMetadata(source=source)
    return SearchResult(document=doc, score=score)


# ---------------------------------------------------------------------------
# ContextWindow
# ---------------------------------------------------------------------------


class TestContextWindow:
    def test_can_add_fits(self):
        window = ContextWindow(max_chars=100)
        doc = Document(content="short")
        assert window.can_add(doc) is True

    def test_can_add_doesnt_fit(self):
        window = ContextWindow(max_chars=5)
        doc = Document(content="too long text")
        assert window.can_add(doc) is False

    def test_add_success(self):
        window = ContextWindow(max_chars=100)
        doc = Document(content="hello")
        assert window.add(doc) is True
        assert len(window.documents) == 1
        assert window.total_chars == 5

    def test_add_fails_when_full(self):
        window = ContextWindow(max_chars=5)
        doc = Document(content="too long")
        assert window.add(doc) is False
        assert len(window.documents) == 0

    def test_get_text(self):
        window = ContextWindow(max_chars=100)
        window.add(Document(content="aaa"))
        window.add(Document(content="bbb"))
        assert window.get_text(separator=" | ") == "aaa | bbb"


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class TestContextBuilder:
    def test_build_relevance(self):
        builder = ContextBuilder(strategy=ContextStrategy.RELEVANCE, max_chars=10000)
        results = [
            _make_result("low score doc", 0.3, "src1"),
            _make_result("high score doc", 0.9, "src2"),
        ]
        text = builder.build(results)
        # Higher score doc should come first
        assert text.index("high score") < text.index("low score")

    def test_build_recency(self):
        from datetime import datetime

        from alpha_zero.utils.compat import UTC

        builder = ContextBuilder(strategy=ContextStrategy.RECENCY, max_chars=10000)
        r1 = _make_result("old doc", 0.9, "src1")
        r1.document.metadata.created_at = datetime(2025, 1, 1, tzinfo=UTC)
        r2 = _make_result("new doc", 0.5, "src2")
        r2.document.metadata.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        text = builder.build([r1, r2])
        assert text.index("new doc") < text.index("old doc")

    def test_build_diversity(self):
        builder = ContextBuilder(strategy=ContextStrategy.DIVERSITY, max_chars=10000)
        results = [
            _make_result("a1", 0.9, "srcA"),
            _make_result("a2", 0.8, "srcA"),
            _make_result("b1", 0.85, "srcB"),
        ]
        text = builder.build(results)
        # Both sources should appear
        assert "a1" in text or "a2" in text
        assert "b1" in text

    def test_build_diversity_empty(self):
        builder = ContextBuilder(strategy=ContextStrategy.DIVERSITY, max_chars=10000)
        text = builder.build([])
        assert text == ""

    def test_build_hybrid(self):
        builder = ContextBuilder(strategy=ContextStrategy.HYBRID, max_chars=10000)
        results = [
            _make_result("srcA doc1", 0.9, "srcA"),
            _make_result("srcA doc2", 0.85, "srcA"),
            _make_result("srcB doc1", 0.8, "srcB"),
        ]
        text = builder.build(results)
        assert len(text) > 0

    def test_build_hybrid_empty(self):
        builder = ContextBuilder(strategy=ContextStrategy.HYBRID, max_chars=10000)
        text = builder.build([])
        assert text == ""

    def test_build_min_score_filter(self):
        builder = ContextBuilder(min_score=0.5, max_chars=10000)
        results = [
            _make_result("good", 0.8, "src1"),
            _make_result("bad", 0.2, "src2"),
        ]
        text = builder.build(results)
        assert "good" in text
        assert "bad" not in text

    def test_build_deduplicate(self):
        builder = ContextBuilder(deduplicate=True, max_chars=10000)
        results = [
            _make_result("first from A", 0.9, "srcA"),
            _make_result("second from A", 0.8, "srcA"),
        ]
        text = builder.build(results)
        assert "first from A" in text
        assert "second from A" not in text

    def test_build_no_deduplicate(self):
        builder = ContextBuilder(deduplicate=False, max_chars=10000)
        results = [
            _make_result("first from A", 0.9, "srcA"),
            _make_result("second from A", 0.8, "srcA"),
        ]
        text = builder.build(results)
        assert "first from A" in text
        assert "second from A" in text

    def test_build_text_char_limit(self):
        builder = ContextBuilder(max_chars=50)
        results = [
            _make_result("a" * 100, 0.9, "src1"),
        ]
        text = builder.build(results)
        assert len(text) <= 100  # truncated with header
        assert "..." in text

    def test_build_text_no_metadata(self):
        builder = ContextBuilder(include_metadata=False, max_chars=10000)
        results = [_make_result("content here", 0.9, "src1")]
        text = builder.build(results)
        assert "Source:" not in text
        assert "content here" in text

    def test_build_text_with_metadata(self):
        builder = ContextBuilder(include_metadata=True, max_chars=10000)
        results = [_make_result("content here", 0.9, "my_source")]
        text = builder.build(results)
        assert "my_source" in text
        assert "0.900" in text

    def test_build_windows(self):
        builder = ContextBuilder()
        results = [
            _make_result("short", 0.9, "s1"),
            _make_result("a" * 5000, 0.8, "s2"),
            _make_result("b" * 5000, 0.7, "s3"),
        ]
        windows = builder.build_windows(results, window_size=6000, overlap=200)
        assert len(windows) >= 2

    def test_build_windows_empty(self):
        builder = ContextBuilder()
        windows = builder.build_windows([], window_size=4000)
        assert windows == []


# ---------------------------------------------------------------------------
# QueryExpansion
# ---------------------------------------------------------------------------


class TestQueryExpansion:
    def test_all_queries(self):
        expansion = QueryExpansion(
            original="test query",
            expanded=["expanded 1", "expanded 2"],
        )
        all_q = expansion.all_queries()
        assert all_q[0] == "test query"
        assert len(all_q) == 3


# ---------------------------------------------------------------------------
# QueryEnhancer
# ---------------------------------------------------------------------------


class TestQueryEnhancer:
    def test_extract_keywords(self):
        enhancer = QueryEnhancer()
        keywords = enhancer.extract_keywords("How to win the early game?")
        assert "win" in keywords
        assert "early" in keywords
        assert "game" in keywords
        # Stop words removed
        assert "how" not in keywords
        assert "the" not in keywords
        assert "to" not in keywords

    def test_extract_keywords_min_length(self):
        enhancer = QueryEnhancer(min_keyword_length=5)
        keywords = enhancer.extract_keywords("How to win the early game?")
        assert "win" not in keywords  # 3 chars
        assert "early" in keywords  # 5 chars

    def test_expand_with_synonyms(self):
        enhancer = QueryEnhancer(use_synonyms=True)
        expansion = enhancer.expand("How to win early game?")
        # Should have synonym-based expansions
        assert len(expansion.expanded) > 0
        # Check that synonyms like "victory" appear
        all_text = " ".join(expansion.expanded)
        assert any(s in all_text for s in ["victory", "succeed", "triumph", "initial", "beginning"])

    def test_expand_no_synonyms(self):
        enhancer = QueryEnhancer(use_synonyms=False)
        expansion = enhancer.expand("How to win early game?")
        # Only keyword-focused query should be generated
        assert len(expansion.expanded) <= 1

    def test_expand_max_expansions(self):
        enhancer = QueryEnhancer(max_expansions=2)
        expansion = enhancer.expand("How to win and attack in early game?")
        assert len(expansion.expanded) <= 2

    def test_expand_keyword_query_added(self):
        enhancer = QueryEnhancer(use_synonyms=False)
        expansion = enhancer.expand("How to win the early game efficiently?")
        # Keywords should form a unique query
        assert len(expansion.keywords) > 0

    def test_decompose_simple(self):
        enhancer = QueryEnhancer()
        sub_queries = enhancer.decompose("simple query")
        assert "simple query" in sub_queries

    def test_decompose_with_and(self):
        enhancer = QueryEnhancer()
        sub_queries = enhancer.decompose("strategy and tactics")
        assert len(sub_queries) >= 2

    def test_decompose_with_question_marks(self):
        enhancer = QueryEnhancer()
        sub_queries = enhancer.decompose("What units? What buildings?")
        assert len(sub_queries) >= 2

    def test_generate_hypothetical(self):
        enhancer = QueryEnhancer()
        hypo = enhancer.generate_hypothetical("How to win early game?")
        assert "discusses" in hypo
        assert "win" in hypo

    def test_generate_hypothetical_no_keywords(self):
        enhancer = QueryEnhancer()
        hypo = enhancer.generate_hypothetical("a")
        # Falls back to original query
        assert hypo == "a"

    def test_normalize(self):
        enhancer = QueryEnhancer()
        result = enhancer.normalize("  How  To  WIN?  ")
        assert result == "how to win?"

    def test_normalize_removes_special_chars(self):
        enhancer = QueryEnhancer()
        result = enhancer.normalize("hello @#$ world!")
        assert "@" not in result
        assert "hello" in result
        assert "world!" in result
