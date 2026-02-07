"""Context building strategies for RAG agents.

This module provides strategies for building context from retrieved documents:
- Relevance-based ranking
- Context window management
- Source deduplication
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from alpha_zero.rag.stores.base import Document, SearchResult


class ContextStrategy(str, Enum):
    """Strategy for building context."""

    RELEVANCE = "relevance"  # Rank by relevance score
    RECENCY = "recency"  # Rank by document date
    DIVERSITY = "diversity"  # Maximize source diversity
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class ContextWindow:
    """A window of context with token/character limits.

    Attributes:
        max_chars: Maximum characters in context
        max_tokens: Maximum tokens (estimated)
        overlap_chars: Overlap between windows
        documents: Documents in this window
        total_chars: Current character count
    """

    max_chars: int = 4000
    max_tokens: int = 1000
    overlap_chars: int = 200
    documents: list[Document] = field(default_factory=list)
    total_chars: int = 0

    def can_add(self, document: Document) -> bool:
        """Check if document can be added to window.

        Args:
            document: Document to add

        Returns:
            True if document fits
        """
        return self.total_chars + len(document.content) <= self.max_chars

    def add(self, document: Document) -> bool:
        """Add document to window if it fits.

        Args:
            document: Document to add

        Returns:
            True if document was added
        """
        if self.can_add(document):
            self.documents.append(document)
            self.total_chars += len(document.content)
            return True
        return False

    def get_text(self, separator: str = "\n\n") -> str:
        """Get combined text from all documents.

        Args:
            separator: Separator between documents

        Returns:
            Combined text
        """
        return separator.join(doc.content for doc in self.documents)


class ContextBuilder:
    """Build context from retrieved documents.

    Provides various strategies for selecting and ordering documents
    to maximize relevance within token/character limits.

    Example:
        builder = ContextBuilder(
            strategy=ContextStrategy.HYBRID,
            max_chars=4000
        )
        context = builder.build(search_results)
    """

    def __init__(
        self,
        strategy: ContextStrategy = ContextStrategy.RELEVANCE,
        max_chars: int = 4000,
        min_score: float = 0.0,
        deduplicate: bool = True,
        include_metadata: bool = True,
    ) -> None:
        """Initialize context builder.

        Args:
            strategy: Context building strategy
            max_chars: Maximum context characters
            min_score: Minimum relevance score to include
            deduplicate: Whether to deduplicate by source
            include_metadata: Whether to include source metadata
        """
        self.strategy = strategy
        self.max_chars = max_chars
        self.min_score = min_score
        self.deduplicate = deduplicate
        self.include_metadata = include_metadata

    def build(self, results: list[SearchResult]) -> str:
        """Build context from search results.

        Args:
            results: Search results from vector store

        Returns:
            Formatted context string
        """
        # Filter by minimum score
        filtered = [r for r in results if r.score >= self.min_score]

        # Apply strategy
        if self.strategy == ContextStrategy.RELEVANCE:
            ordered = self._rank_by_relevance(filtered)
        elif self.strategy == ContextStrategy.RECENCY:
            ordered = self._rank_by_recency(filtered)
        elif self.strategy == ContextStrategy.DIVERSITY:
            ordered = self._rank_by_diversity(filtered)
        else:
            ordered = self._rank_hybrid(filtered)

        # Deduplicate if enabled
        if self.deduplicate:
            ordered = self._deduplicate(ordered)

        # Build context within character limit
        return self._build_text(ordered)

    def _rank_by_relevance(self, results: list[SearchResult]) -> list[SearchResult]:
        """Rank results by relevance score.

        Args:
            results: Search results

        Returns:
            Sorted results
        """
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _rank_by_recency(self, results: list[SearchResult]) -> list[SearchResult]:
        """Rank results by document date.

        Args:
            results: Search results

        Returns:
            Sorted results
        """
        return sorted(
            results,
            key=lambda r: r.document.metadata.created_at,
            reverse=True,
        )

    def _rank_by_diversity(self, results: list[SearchResult]) -> list[SearchResult]:
        """Rank results to maximize source diversity.

        Args:
            results: Search results

        Returns:
            Reordered results
        """
        if not results:
            return results

        # Group by source
        by_source: dict[str, list[SearchResult]] = {}
        for result in results:
            source = result.document.metadata.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)

        # Interleave sources
        ordered = []
        source_iters = {
            source: iter(sorted(items, key=lambda r: r.score, reverse=True))
            for source, items in by_source.items()
        }

        while source_iters:
            exhausted = []
            for source, it in source_iters.items():
                try:
                    ordered.append(next(it))
                except StopIteration:
                    exhausted.append(source)
            for source in exhausted:
                del source_iters[source]

        return ordered

    def _rank_hybrid(self, results: list[SearchResult]) -> list[SearchResult]:
        """Hybrid ranking combining relevance and diversity.

        Args:
            results: Search results

        Returns:
            Reordered results
        """
        if not results:
            return results

        # Score combining relevance and diversity
        seen_sources: set[str] = set()
        scored = []

        for result in sorted(results, key=lambda r: r.score, reverse=True):
            source = result.document.metadata.source
            diversity_bonus = 0.2 if source not in seen_sources else 0.0
            hybrid_score = result.score + diversity_bonus
            scored.append((hybrid_score, result))
            seen_sources.add(source)

        return [r for _, r in sorted(scored, key=lambda x: x[0], reverse=True)]

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate documents by source.

        Args:
            results: Search results

        Returns:
            Deduplicated results
        """
        seen_sources: set[str] = set()
        deduped = []

        for result in results:
            source = result.document.metadata.source
            if source not in seen_sources:
                deduped.append(result)
                seen_sources.add(source)

        return deduped

    def _build_text(self, results: list[SearchResult]) -> str:
        """Build context text within character limit.

        Args:
            results: Ordered search results

        Returns:
            Formatted context string
        """
        parts: list[str] = []
        total_chars = 0

        for i, result in enumerate(results, start=1):
            doc = result.document
            content = doc.content

            # Check if we can add this document
            doc_chars = len(content)
            if self.include_metadata:
                source = doc.metadata.source or "Unknown"
                header = f"[{i}] Source: {source} (score: {result.score:.3f})\n"
                doc_chars += len(header)
            else:
                header = f"[{i}] "

            if total_chars + doc_chars > self.max_chars:
                # Truncate if this is the first document
                if not parts:
                    remaining = self.max_chars - len(header)
                    content = content[:remaining] + "..."
                    parts.append(header + content)
                break

            parts.append(header + content)
            total_chars += doc_chars

        return "\n\n---\n\n".join(parts)

    def build_windows(
        self,
        results: list[SearchResult],
        window_size: int = 4000,
        overlap: int = 200,
    ) -> list[ContextWindow]:
        """Build multiple context windows for long content.

        Args:
            results: Search results
            window_size: Characters per window
            overlap: Overlap between windows

        Returns:
            List of context windows
        """
        windows = []
        current_window = ContextWindow(max_chars=window_size, overlap_chars=overlap)

        for result in self._rank_by_relevance(results):
            doc = result.document
            if current_window.can_add(doc):
                current_window.add(doc)
            else:
                if current_window.documents:
                    windows.append(current_window)
                current_window = ContextWindow(max_chars=window_size, overlap_chars=overlap)
                current_window.add(doc)

        if current_window.documents:
            windows.append(current_window)

        return windows
