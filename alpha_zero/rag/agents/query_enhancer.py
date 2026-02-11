"""Query enhancement for RAG retrieval.

This module provides query enhancement techniques:
- Query expansion with synonyms
- Query decomposition
- Hypothetical document generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class QueryExpansion:
    """Result of query expansion.

    Attributes:
        original: Original query
        expanded: Expanded queries
        keywords: Extracted keywords
        metadata: Additional metadata
    """

    original: str
    expanded: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_queries(self) -> list[str]:
        """Get all queries including original.

        Returns:
            List of all queries
        """
        return [self.original, *self.expanded]


class QueryEnhancer:
    """Enhance queries for better retrieval.

    Provides various query enhancement techniques to improve
    retrieval quality from vector stores.

    Example:
        enhancer = QueryEnhancer()
        expansion = enhancer.expand("How to win early game?")
        print(expansion.all_queries())
    """

    # Common stop words to filter
    STOP_WORDS: ClassVar[set[str]] = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    }

    # Domain-specific synonyms (can be expanded)
    SYNONYMS: ClassVar[dict[str, list[str]]] = {
        "win": ["victory", "succeed", "triumph"],
        "lose": ["defeat", "fail", "lost"],
        "strategy": ["tactic", "approach", "plan"],
        "early": ["initial", "beginning", "start"],
        "late": ["end", "final", "conclusion"],
        "game": ["match", "round"],
        "build": ["construct", "create", "develop"],
        "attack": ["assault", "offense", "strike"],
        "defend": ["protect", "guard", "defense"],
        "unit": ["troop", "soldier", "military"],
        "city": ["settlement", "town"],
        "resource": ["material", "supply"],
    }

    def __init__(
        self,
        use_synonyms: bool = True,
        max_expansions: int = 3,
        min_keyword_length: int = 3,
    ) -> None:
        """Initialize query enhancer.

        Args:
            use_synonyms: Whether to expand with synonyms
            max_expansions: Maximum number of expanded queries
            min_keyword_length: Minimum keyword length
        """
        self.use_synonyms = use_synonyms
        self.max_expansions = max_expansions
        self.min_keyword_length = min_keyword_length

    def expand(self, query: str) -> QueryExpansion:
        """Expand a query with variations.

        Args:
            query: Original query

        Returns:
            QueryExpansion with variations
        """
        # Extract keywords
        keywords = self.extract_keywords(query)

        # Generate expansions
        expansions = []

        if self.use_synonyms:
            synonym_queries = self._expand_with_synonyms(query, keywords)
            expansions.extend(synonym_queries[: self.max_expansions])

        # Add keyword-focused query
        if keywords:
            keyword_query = " ".join(keywords)
            if keyword_query != query.lower() and keyword_query not in expansions:
                expansions.append(keyword_query)

        return QueryExpansion(
            original=query,
            expanded=expansions[: self.max_expansions],
            keywords=keywords,
        )

    def extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Tokenize and clean
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        # Filter stop words and short words
        keywords = [
            word
            for word in words
            if word not in self.STOP_WORDS and len(word) >= self.min_keyword_length
        ]

        return keywords

    def _expand_with_synonyms(
        self,
        query: str,
        keywords: list[str],
    ) -> list[str]:
        """Expand query using synonyms.

        Args:
            query: Original query
            keywords: Extracted keywords

        Returns:
            List of expanded queries
        """
        expansions = []

        for keyword in keywords:
            if keyword in self.SYNONYMS:
                for synonym in self.SYNONYMS[keyword]:
                    # Use word boundaries to avoid replacing substrings within words
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    expanded = re.sub(pattern, synonym, query.lower())
                    if expanded != query.lower():
                        expansions.append(expanded)

        return expansions

    def decompose(self, query: str) -> list[str]:
        """Decompose a complex query into sub-queries.

        Args:
            query: Complex query

        Returns:
            List of sub-queries
        """
        sub_queries = [query]

        # Split on common conjunctions
        if " and " in query.lower():
            parts = re.split(r"\s+and\s+", query, flags=re.IGNORECASE)
            sub_queries.extend(parts)

        # Split on question words
        if "?" in query:
            questions = re.split(r"\?\s*", query)
            sub_queries.extend([q.strip() + "?" for q in questions if q.strip()])

        return list(set(sub_queries))

    def generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document for HyDE retrieval.

        This creates a hypothetical answer that would be similar
        to documents in the collection.

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        # Simple template-based generation
        keywords = self.extract_keywords(query)

        if not keywords:
            return query

        # Generate a simple hypothetical document
        keyword_str = ", ".join(keywords)
        return f"This document discusses {keyword_str}. {query}"

    def normalize(self, query: str) -> str:
        """Normalize a query for consistent processing.

        Args:
            query: Input query

        Returns:
            Normalized query
        """
        # Lowercase
        query = query.lower()

        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query).strip()

        # Remove special characters except punctuation
        query = re.sub(r"[^\w\s\?\.\,\!]", "", query)

        return query
