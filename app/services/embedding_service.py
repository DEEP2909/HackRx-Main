from typing import List, Dict
from loguru import logger
import httpx
import asyncio
from app.core.config import settings
from app.services.cache_service import cache_service

class EmbeddingService:
    """Embedding service with intelligent caching to reduce API calls and improve response times"""
    
    def __init__(self):
        self.model_name = settings.OPENAI_EMBEDDING_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.request_delay = 0.1  # Delay between requests to avoid rate limits
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'api_requests': 0,
            'tokens_saved': 0
        }

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with smart caching to minimize API calls"""
        logger.info(f"Fetching embeddings for {len(texts)} texts")
        
        # Check cache for existing embeddings
        cached_embeddings = await cache_service.get_cached_embeddings(texts)
        
        # Separate cached and non-cached texts
        cached_texts = set(cached_embeddings.keys())
        non_cached_texts = [text for text in texts if text not in cached_texts]
        
        self.stats['total_requests'] += len(texts)
        self.stats['cached_requests'] += len(cached_texts)
        self.stats['api_requests'] += len(non_cached_texts)
        
        if cached_texts:
            logger.info(f"Found {len(cached_texts)} cached embeddings, fetching {len(non_cached_texts)} from API")
        
        # Fetch non-cached embeddings from API
        new_embeddings = {}
        if non_cached_texts:
            try:
                # Process in batches to avoid API limits
                api_embeddings = await self._fetch_embeddings_in_batches(non_cached_texts)
                new_embeddings = dict(zip(non_cached_texts, api_embeddings))
                
                # Cache the new embeddings
                await cache_service.cache_embeddings(non_cached_texts, api_embeddings, ttl=3600)
                logger.info(f"Cached {len(new_embeddings)} new embeddings")
                
            except Exception as e:
                logger.error(f"Error fetching embeddings from API: {e}")
                raise
        
        # Combine cached and new embeddings in original order
        result_embeddings = []
        for text in texts:
            if text in cached_embeddings:
                result_embeddings.append(cached_embeddings[text])
            elif text in new_embeddings:
                result_embeddings.append(new_embeddings[text])
            else:
                logger.error(f"No embedding found for text: {text[:50]}...")
                # Return zero vector as fallback
                result_embeddings.append([0.0] * 1536)  # Ada-002 dimension
        
        logger.info(f"Successfully retrieved {len(result_embeddings)} embeddings ({len(cached_texts)} from cache)")
        return result_embeddings
    
    async def _fetch_embeddings_in_batches(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings in batches to respect API limits"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Fetching batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1}")
            
            try:
                batch_embeddings = await self._fetch_single_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Add delay between batches to respect rate limits
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error fetching batch {i // self.batch_size + 1}: {e}")
                raise
        
        return all_embeddings
    
    async def _fetch_single_batch(self, texts: List[str]) -> List[List[float]]:
        """Fetch a single batch of embeddings from OpenAI API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_name,
            'input': texts,
            'encoding_format': 'float'
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings_data = result.get('data', [])
            
            # Extract embeddings in correct order
            embeddings = []
            for embedding_obj in sorted(embeddings_data, key=lambda x: x['index']):
                embeddings.append(embedding_obj['embedding'])
            
            return embeddings
    
    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with caching"""
        embeddings = await self.get_embeddings([text])
        return embeddings[0]
    
    async def precompute_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        """Precompute and cache embeddings for a list of texts"""
        logger.info(f"Precomputing embeddings for {len(texts)} texts")
        
        # Filter out already cached texts
        cached_embeddings = await cache_service.get_cached_embeddings(texts)
        non_cached_texts = [text for text in texts if text not in cached_embeddings]
        
        if not non_cached_texts:
            logger.info("All embeddings already cached")
            return cached_embeddings
        
        # Fetch new embeddings
        new_embeddings_list = await self._fetch_embeddings_in_batches(non_cached_texts)
        new_embeddings = dict(zip(non_cached_texts, new_embeddings_list))
        
        # Cache new embeddings with longer TTL for precomputed ones
        await cache_service.cache_embeddings(non_cached_texts, new_embeddings_list, ttl=7200)
        
        # Combine all embeddings
        all_embeddings = {**cached_embeddings, **new_embeddings}
        
        logger.info(f"Precomputed {len(new_embeddings)} new embeddings, total: {len(all_embeddings)}")
        return all_embeddings
    
    async def warm_up_cache(self, common_texts: List[str]):
        """Warm up cache with commonly used texts"""
        logger.info(f"Warming up embedding cache with {len(common_texts)} common texts")
        await self.precompute_embeddings(common_texts)
    
    async def get_embedding_stats(self) -> Dict:
        """Get embedding service statistics"""
        cache_stats = await cache_service.get_cache_stats()
        
        return {
            'total_requests': self.stats['total_requests'],
            'cached_requests': self.stats['cached_requests'],
            'api_requests': self.stats['api_requests'],
            'cache_hit_rate': (self.stats['cached_requests'] / max(self.stats['total_requests'], 1)) * 100,
            'embedding_cache_entries': cache_stats['entries_by_type'].get('embeddings', 0),
            'tokens_saved_estimate': self.stats['cached_requests'] * 100  # Rough estimate
        }
    
    async def clear_embedding_cache(self):
        """Clear embedding cache"""
        await cache_service.clear_cache(cache_service.embedding_cache_prefix)
        logger.info("Cleared embedding cache")

# Singleton instance
embedding_service = EmbeddingService()
