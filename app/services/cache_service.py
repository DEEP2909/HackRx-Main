import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from loguru import logger
import asyncio
from functools import wraps

from app.core.config import settings
from app.models.document import DocumentChunk

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = None
    size_bytes: int = 0

class CacheService:
    """Multi-level cache service for documents, embeddings, and query results"""
    
    def __init__(self):
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_items = 1000
        self.default_ttl = settings.CACHE_TTL_SECONDS
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Cache categories
        self.document_cache_prefix = "doc:"
        self.embedding_cache_prefix = "emb:"
        self.query_cache_prefix = "query:"
        self.chunk_cache_prefix = "chunk:"
        
    def _generate_cache_key(self, prefix: str, data: Union[str, Dict, List]) -> str:
        """Generate a consistent cache key"""
        if isinstance(data, str):
            key_data = data
        else:
            key_data = json.dumps(data, sort_keys=True)
        
        hash_object = hashlib.md5(key_data.encode())
        return f"{prefix}{hash_object.hexdigest()}"
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate the memory size of a value"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, dict)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            elif isinstance(value, DocumentChunk):
                return len(value.content.encode('utf-8')) + 1000  # estimate for metadata
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 1000  # fallback estimate
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        self.stats['total_requests'] += 1
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check if expired
            if time.time() - entry.created_at > entry.ttl:
                await self.delete(key)
                self.stats['misses'] += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            self.stats['hits'] += 1
            logger.debug(f"Cache HIT for key: {key[:20]}...")
            return entry.value
        
        self.stats['misses'] += 1
        logger.debug(f"Cache MISS for key: {key[:20]}...")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set item in cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        # Calculate size
        size_bytes = self._calculate_size(value)
        
        # Check if we need to evict items
        if len(self.memory_cache) >= self.max_memory_items:
            await self._evict_lru()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl,
            size_bytes=size_bytes,
            last_accessed=time.time()
        )
        
        self.memory_cache[key] = entry
        logger.debug(f"Cache SET for key: {key[:20]}... (size: {size_bytes} bytes)")
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache"""
        if key in self.memory_cache:
            del self.memory_cache[key]
            logger.debug(f"Cache DELETE for key: {key[:20]}...")
            return True
        return False
    
    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self.memory_cache:
            return
        
        # Find LRU item
        lru_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k].last_accessed or 0
        )
        
        await self.delete(lru_key)
        self.stats['evictions'] += 1
        logger.debug(f"Evicted LRU item: {lru_key[:20]}...")
    
    # Document-specific cache methods
    async def cache_document_chunks(self, url: str, chunks: List[DocumentChunk], ttl: Optional[float] = None):
        """Cache processed document chunks"""
        key = self._generate_cache_key(self.document_cache_prefix, url)
        
        # Convert chunks to serializable format
        serializable_chunks = []
        for chunk in chunks:
            chunk_data = {
                'content': chunk.content,
                'metadata': chunk.metadata,
                'embedding': chunk.embedding,
                'chunk_id': chunk.chunk_id,
                'created_at': chunk.created_at.isoformat() if chunk.created_at else None
            }
            serializable_chunks.append(chunk_data)
        
        await self.set(key, serializable_chunks, ttl or 7200)  # 2 hours for documents
        logger.info(f"Cached {len(chunks)} chunks for document: {url}")
    
    async def get_document_chunks(self, url: str) -> Optional[List[DocumentChunk]]:
        """Get cached document chunks"""
        key = self._generate_cache_key(self.document_cache_prefix, url)
        cached_data = await self.get(key)
        
        if cached_data:
            # Convert back to DocumentChunk objects
            chunks = []
            for chunk_data in cached_data:
                chunk = DocumentChunk(
                    content=chunk_data['content'],
                    metadata=chunk_data['metadata'],
                    embedding=chunk_data.get('embedding'),
                    chunk_id=chunk_data.get('chunk_id')
                )
                chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} cached chunks for document: {url}")
            return chunks
        
        return None
    
    # Embedding cache methods
    async def cache_embeddings(self, texts: List[str], embeddings: List[List[float]], ttl: Optional[float] = None):
        """Cache text embeddings"""
        for text, embedding in zip(texts, embeddings):
            key = self._generate_cache_key(self.embedding_cache_prefix, text)
            await self.set(key, embedding, ttl or 3600)  # 1 hour for embeddings
        
        logger.info(f"Cached {len(embeddings)} embeddings")
    
    async def get_cached_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        """Get cached embeddings for texts"""
        cached_embeddings = {}
        
        for text in texts:
            key = self._generate_cache_key(self.embedding_cache_prefix, text)
            embedding = await self.get(key)
            if embedding:
                cached_embeddings[text] = embedding
        
        if cached_embeddings:
            logger.info(f"Found {len(cached_embeddings)} cached embeddings out of {len(texts)} requested")
        
        return cached_embeddings
    
    # Query result cache methods
    async def cache_query_result(self, question: str, document_url: str, answer: str, ttl: Optional[float] = None):
        """Cache query result"""
        cache_data = {
            'question': question,
            'document_url': document_url,
            'timestamp': time.time()
        }
        key = self._generate_cache_key(self.query_cache_prefix, cache_data)
        
        result_data = {
            'answer': answer,
            'cached_at': time.time(),
            'question': question,
            'document_url': document_url
        }
        
        await self.set(key, result_data, ttl or 1800)  # 30 minutes for query results
        logger.info(f"Cached query result for: {question[:50]}...")
    
    async def get_cached_query_result(self, question: str, document_url: str) -> Optional[str]:
        """Get cached query result"""
        cache_data = {
            'question': question,
            'document_url': document_url,
            'timestamp': time.time()
        }
        key = self._generate_cache_key(self.query_cache_prefix, cache_data)
        
        # Try with current timestamp (exact match unlikely)
        result = await self.get(key)
        if result:
            return result['answer']
        
        # Try to find similar query (search all query cache keys)
        for cached_key in self.memory_cache.keys():
            if cached_key.startswith(self.query_cache_prefix):
                cached_result = self.memory_cache[cached_key].value
                if (cached_result.get('question') == question and 
                    cached_result.get('document_url') == document_url):
                    logger.info(f"Found cached query result for: {question[:50]}...")
                    return cached_result['answer']
        
        return None
    
    # Utility methods
    async def clear_cache(self, prefix: Optional[str] = None):
        """Clear cache entries, optionally by prefix"""
        if prefix:
            keys_to_delete = [key for key in self.memory_cache.keys() if key.startswith(prefix)]
            for key in keys_to_delete:
                await self.delete(key)
            logger.info(f"Cleared {len(keys_to_delete)} cache entries with prefix: {prefix}")
        else:
            self.memory_cache.clear()
            logger.info("Cleared all cache entries")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(entry.size_bytes for entry in self.memory_cache.values())
        
        return {
            'total_entries': len(self.memory_cache),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'hit_rate': self.stats['hits'] / max(self.stats['total_requests'], 1),
            'statistics': self.stats,
            'entries_by_type': {
                'documents': len([k for k in self.memory_cache.keys() if k.startswith(self.document_cache_prefix)]),
                'embeddings': len([k for k in self.memory_cache.keys() if k.startswith(self.embedding_cache_prefix)]),
                'queries': len([k for k in self.memory_cache.keys() if k.startswith(self.query_cache_prefix)]),
            }
        }
    
    async def cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.memory_cache.items():
            if current_time - entry.created_at > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    # Background cleanup task
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await self.cleanup_expired()
                    await asyncio.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
        
        asyncio.create_task(cleanup_loop())
        logger.info("Started cache cleanup background task")

# Cache decorator
def cached(prefix: str = "", ttl: Optional[float] = None):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and args
            cache_key_data = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key = cache_service._generate_cache_key(prefix, cache_key_data)
            
            # Try to get from cache
            result = await cache_service.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_service.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Singleton instance
cache_service = CacheService()
