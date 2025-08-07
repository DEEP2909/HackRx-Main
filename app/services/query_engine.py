import time
from typing import List, Dict, Any
from loguru import logger
import asyncio

from app.services.document_processor import document_processor
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store_service
from app.services.llm_service import llm_service
from app.services.cache_service import cache_service
from app.core.config import settings

class QueryEngine:
    """Main query processing engine with comprehensive caching for faster responses"""
    
    def __init__(self):
        self.processing_stats = {
            'total_queries': 0,
            'cached_answers': 0,
            'processing_times': [],
            'document_cache_hits': 0
        }
        
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process a query with document URL and questions using multi-level caching"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query for document: {document_url} with {len(questions)} questions")
            self.processing_stats['total_queries'] += 1
            
            # Step 1: Check for fully cached query results first
            cached_answers = await self._check_cached_answers(document_url, questions)
            if len(cached_answers) == len(questions):
                processing_time = time.time() - start_time
                logger.info(f"All answers found in cache! Processing time: {processing_time:.2f}s")
                return cached_answers
            
            # Step 2: Process document if not cached
            await self._ensure_document_processed(document_url)
            
            # Step 3: Process questions (mix of cached and new)
            answers = await self._process_questions_with_cache(document_url, questions, cached_answers)
            
            # Log processing statistics
            processing_time = time.time() - start_time
            self.processing_stats['processing_times'].append(processing_time)
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            raise
    
    async def _check_cached_answers(self, document_url: str, questions: List[str]) -> List[str]:
        """Check cache for existing answers to questions"""
        cached_answers = []
        
        for question in questions:
            cached_answer = await cache_service.get_cached_query_result(question, document_url)
            if cached_answer:
                cached_answers.append(cached_answer)
                self.processing_stats['cached_answers'] += 1
                logger.debug(f"Found cached answer for: {question[:50]}...")
            else:
                cached_answers.append(None)
        
        # Return only the non-None answers
        valid_answers = [ans for ans in cached_answers if ans is not None]
        
        if valid_answers:
            logger.info(f"Found {len(valid_answers)} cached answers out of {len(questions)} questions")
        
        return valid_answers
    
    async def _ensure_document_processed(self, document_url: str):
        """Ensure document is processed and indexed with caching"""
        
        # Check if document chunks are cached
        cached_chunks = await cache_service.get_document_chunks(document_url)
        if cached_chunks:
            logger.info(f"Document found in cache with {len(cached_chunks)} chunks")
            self.processing_stats['document_cache_hits'] += 1
            
            # Verify chunks are in vector store
            sample_embedding = cached_chunks[0].embedding if cached_chunks[0].embedding else None
            if sample_embedding:
                # Quick check if embeddings exist in vector store
                search_results = await vector_store_service.search(sample_embedding, top_k=1)
                if search_results:
                    logger.info("Document already indexed in vector store")
                    return
        
        # Process document if not fully cached/indexed
        logger.info("Processing document (not in cache or vector store)")
        chunks = await document_processor.process_document(document_url)
        
        # Check if any embeddings are cached
        chunk_texts = [chunk.content for chunk in chunks]
        cached_embeddings = await cache_service.get_cached_embeddings(chunk_texts)
        
        # Generate embeddings only for non-cached chunks
        non_cached_texts = [text for text in chunk_texts if text not in cached_embeddings]
        
        if non_cached_texts:
            logger.info(f"Generating embeddings for {len(non_cached_texts)} new chunks")
            new_embeddings_list = await embedding_service.get_embeddings(non_cached_texts)
            new_embeddings = dict(zip(non_cached_texts, new_embeddings_list))
        else:
            new_embeddings = {}
        
        # Assign embeddings to chunks
        for chunk in chunks:
            if chunk.content in cached_embeddings:
                chunk.embedding = cached_embeddings[chunk.content]
            elif chunk.content in new_embeddings:
                chunk.embedding = new_embeddings[chunk.content]
        
        # Add to vector store
        await vector_store_service.add_documents(chunks)
        
        logger.info(f"Document processed and indexed: {len(chunks)} chunks")
    
    async def _process_questions_with_cache(self, document_url: str, questions: List[str], cached_answers: List[str]) -> List[str]:
        """Process questions with intelligent caching"""
        final_answers = []
        
        # Identify which questions need processing
        questions_to_process = []
        answer_indices = {}  # Map question index to final answer index
        
        cached_index = 0
        for i, question in enumerate(questions):
            # Check if we have a cached answer for this question
            if cached_index < len(cached_answers) and cached_answers[cached_index] is not None:
                cached_answer = await cache_service.get_cached_query_result(question, document_url)
                if cached_answer:
                    final_answers.append(cached_answer)
                    cached_index += 1
                    continue
            
            # Need to process this question
            questions_to_process.append(question)
            answer_indices[len(questions_to_process) - 1] = len(final_answers)
            final_answers.append(None)  # Placeholder
        
        # Process non-cached questions
        if questions_to_process:
            logger.info(f"Processing {len(questions_to_process)} non-cached questions")
            new_answers = await self._process_questions_batch(questions_to_process)
            
            # Insert new answers in correct positions
            for i, answer in enumerate(new_answers):
                final_answer_index = answer_indices[i]
                final_answers[final_answer_index] = answer
                
                # Cache the new answer
                question = questions_to_process[i]
                await cache_service.cache_query_result(question, document_url, answer, ttl=1800)
        
        return final_answers
    
    async def _process_questions_batch(self, questions: List[str]) -> List[str]:
        """Process a batch of questions efficiently"""
        
        # Limit concurrent processing to avoid resource exhaustion
        semaphore = asyncio.Semaphore(settings.PARALLEL_QUESTIONS)
        
        async def process_with_semaphore(question):
            async with semaphore:
                return await self._process_single_question(question)
        
        # Process questions concurrently
        tasks = [process_with_semaphore(question) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {i}: {result}")
                answers.append(f"Error processing question: {str(result)}")
            else:
                answers.append(result)
        
        return answers
    
    async def _process_single_question(self, question: str) -> str:
        """Process a single question with embedding caching"""
        logger.info(f"Processing question: {question}")
        
        try:
            # Step 1: Get embedding for question (with caching)
            question_embedding = await embedding_service.get_single_embedding(question)
            
            # Step 2: Search for relevant chunks
            search_results = await vector_store_service.search(
                query_embedding=question_embedding,
                top_k=settings.TOP_K_RESULTS
            )
            
            if not search_results:
                return "No relevant information found in the document to answer this question."
            
            # Step 3: Generate answer using LLM
            llm_response = await llm_service.generate_answer(question, search_results)
            
            # Extract the answer
            answer = llm_response.get('answer', 'Unable to generate answer')
            
            # Log token usage
            token_usage = llm_response.get('token_usage', {})
            logger.info(f"Token usage - Total: {token_usage.get('total_tokens', 0)}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return f"Error processing question: {str(e)}"
    
    async def warm_up_system(self, document_urls: List[str], common_questions: List[str]):
        """Warm up the system by preloading common documents and questions"""
        logger.info(f"Warming up system with {len(document_urls)} documents and {len(common_questions)} questions")
        
        # Preprocess common documents
        for url in document_urls:
            try:
                await self._ensure_document_processed(url)
                logger.info(f"Warmed up document: {url}")
            except Exception as e:
                logger.warning(f"Failed to warm up document {url}: {e}")
        
        # Precompute embeddings for common questions
        if common_questions:
            await embedding_service.precompute_embeddings(common_questions)
        
        logger.info("System warm-up completed")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        cache_stats = await cache_service.get_cache_stats()
        embedding_stats = await embedding_service.get_embedding_stats()
        
        avg_processing_time = (
            sum(self.processing_stats['processing_times']) / 
            len(self.processing_stats['processing_times'])
        ) if self.processing_stats['processing_times'] else 0
        
        return {
            'query_processing': {
                'total_queries': self.processing_stats['total_queries'],
                'cached_answers': self.processing_stats['cached_answers'],
                'document_cache_hits': self.processing_stats['document_cache_hits'],
                'average_processing_time': round(avg_processing_time, 2),
                'cache_hit_rate': (
                    self.processing_stats['cached_answers'] / 
                    max(self.processing_stats['total_queries'], 1)
                ) * 100
            },
            'cache_stats': cache_stats,
            'embedding_stats': embedding_stats
        }
    
    async def clear_all_caches(self):
        """Clear all caches"""
        await cache_service.clear_cache()
        logger.info("Cleared all caches")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            'status': 'healthy',
            'components': {}
        }
        
        try:
            # Check cache service
            cache_stats = await cache_service.get_cache_stats()
            health['components']['cache'] = {
                'status': 'healthy',
                'entries': cache_stats['total_entries'],
                'size_mb': cache_stats['total_size_mb']
            }
        except Exception as e:
            health['components']['cache'] = {'status': 'unhealthy', 'error': str(e)}
            health['status'] = 'degraded'
        
        try:
            # Check vector store
            health['components']['vector_store'] = {'status': 'healthy'}
        except Exception as e:
            health['components']['vector_store'] = {'status': 'unhealthy', 'error': str(e)}
            health['status'] = 'degraded'
        
        return health

# Singleton instance
query_engine = QueryEngine()
