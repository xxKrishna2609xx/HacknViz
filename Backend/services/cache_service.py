import os
import threading
from typing import Dict, Tuple, List, Optional, Any

class CacheService:
    def __init__(self):
        # Key: file path (str) -> Value: Tuple[float, List[float]] (mtime, embedding)
        self._embeddings: Dict[str, Tuple[float, List[float]]] = {}
        # Key: file path (str) -> Value: Tuple[float, str] (mtime, description)
        self._descriptions: Dict[str, Tuple[float, str]] = {}
        
        # Locks for thread-safety
        self._embedding_lock = threading.Lock()
        self._description_lock = threading.Lock()

    def get_embedding(self, filepath: str) -> Optional[List[float]]:
        """Retrieve embedding if cache is valid (file modification time matches)."""
        if not os.path.exists(filepath):
            return None
            
        mtime = os.path.getmtime(filepath)
        with self._embedding_lock:
            cached = self._embeddings.get(filepath)
            if cached and cached[0] == mtime:
                return cached[1]
        return None

    def set_embedding(self, filepath: str, embedding: List[float]) -> None:
        """Store embedding in cache with current file modification time."""
        if not os.path.exists(filepath):
            return
            
        mtime = os.path.getmtime(filepath)
        with self._embedding_lock:
            self._embeddings[filepath] = (mtime, embedding)

    def get_description(self, filepath: str) -> Optional[str]:
        """Retrieve description if cache is valid (file modification time matches)."""
        if not os.path.exists(filepath):
            return None
            
        mtime = os.path.getmtime(filepath)
        with self._description_lock:
            cached = self._descriptions.get(filepath)
            if cached and cached[0] == mtime:
                return cached[1]
        return None

    def set_description(self, filepath: str, description: str) -> None:
        """Store description in cache with current file modification time."""
        if not os.path.exists(filepath):
            return
            
        mtime = os.path.getmtime(filepath)
        with self._description_lock:
            self._descriptions[filepath] = (mtime, description)

    def clear(self) -> None:
        """Clear all in-memory caches."""
        with self._embedding_lock:
            self._embeddings.clear()
        with self._description_lock:
            self._descriptions.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get the count of items in caches."""
        return {
            "cached_embeddings": len(self._embeddings),
            "cached_descriptions": len(self._descriptions)
        }

# Global cache service instance
cache_service = CacheService()
