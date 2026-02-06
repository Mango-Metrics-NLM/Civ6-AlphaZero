"""Caching utilities for RAG system.

Provides:
- LRU cache with size limit
- TTL cache with time-based expiration
- Cache interface for custom implementations
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Cache(ABC, Generic[K, V]):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: K) -> V | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """

    @abstractmethod
    def set(self, key: K, value: V) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """

    @abstractmethod
    def delete(self, key: K) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""

    @abstractmethod
    def size(self) -> int:
        """Get number of items in cache.

        Returns:
            Number of cached items
        """

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache."""
        return self.get(key) is not None


class LRUCache(Cache[K, V]):
    """Least Recently Used (LRU) cache.

    Evicts least recently used items when capacity is exceeded.

    Example:
        cache = LRUCache(maxsize=100)
        cache.set("key", "value")
        value = cache.get("key")
    """

    def __init__(self, maxsize: int = 128) -> None:
        """Initialize LRU cache.

        Args:
            maxsize: Maximum number of items
        """
        self.maxsize = maxsize
        self._cache: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        """Get value and move to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: K, value: V) -> None:
        """Set value and evict LRU if needed.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value

        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def delete(self, key: K) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)


@dataclass
class TTLEntry(Generic[V]):
    """Entry with time-to-live."""

    value: V
    expires_at: float


class TTLCache(Cache[K, V]):
    """Time-to-Live (TTL) cache.

    Items expire after a specified duration.

    Example:
        cache = TTLCache(ttl=300)  # 5 minute TTL
        cache.set("key", "value")
        # After 5 minutes, key will be None
        value = cache.get("key")
    """

    def __init__(self, ttl: float = 300.0, maxsize: int | None = None) -> None:
        """Initialize TTL cache.

        Args:
            ttl: Time-to-live in seconds
            maxsize: Optional maximum size (None for unlimited)
        """
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache: dict[K, TTLEntry[V]] = {}

    def get(self, key: K) -> V | None:
        """Get value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        entry = self._cache.get(key)
        if entry is None:
            return None

        if time.time() > entry.expires_at:
            del self._cache[key]
            return None

        return entry.value

    def set(self, key: K, value: V, ttl: float | None = None) -> None:
        """Set value with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL for this entry
        """
        # Clean up expired entries first
        self._cleanup()

        expires_at = time.time() + (ttl if ttl is not None else self.ttl)
        self._cache[key] = TTLEntry(value=value, expires_at=expires_at)

        # Evict if over maxsize
        if self.maxsize and len(self._cache) > self.maxsize:
            self._evict_oldest()

    def delete(self, key: K) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def size(self) -> int:
        """Get number of non-expired items."""
        self._cleanup()
        return len(self._cache)

    def _cleanup(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now > v.expires_at]
        for key in expired:
            del self._cache[key]

    def _evict_oldest(self) -> None:
        """Evict oldest entry by expiration time."""
        if not self._cache:
            return

        oldest_key = min(self._cache, key=lambda k: self._cache[k].expires_at)
        del self._cache[oldest_key]


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Hash string for cache key
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()
