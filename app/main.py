from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.api.routes import query_router
from app.services.vector_store import vector_store_service
from app.services.cache_service import cache_service

# Initialize services on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM Query Retrieval System...")
    
    # Initialize cache service first
    logger.info("Initializing cache service...")
    await cache_service.start_cleanup_task()
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = vector_store_service
    await vector_store.initialize()
    
    # Log startup statistics
    cache_stats = await cache_service.get_cache_stats()
    logger.info(f"Cache service initialized: {cache_stats['total_entries']} entries")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Query Retrieval System...")
    
    # Cleanup cache service
    await cache_service.cleanup_expired()
    cache_stats = await cache_service.get_cache_stats()
    logger.info(f"Cache service shutdown: {cache_stats['total_entries']} entries retained")
    
    # Close vector store
    await vector_store.close()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="LLM-Powered Intelligent Query-Retrieval System with Advanced Caching",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    query_router,
    prefix=f"{settings.API_PREFIX}/hackrx",
    tags=["query"]
)

@app.get("/")
async def root(request: Request):
    base_url = str(request.base_url)
    cache_stats = await cache_service.get_cache_stats()
    
    return {
        "message": "LLM Query Retrieval System API with Advanced Caching",
        "version": "1.0.0",
        "docs": f"{base_url}docs",
        "cache_status": {
            "entries": cache_stats['total_entries'],
            "size_mb": cache_stats['total_size_mb'],
            "hit_rate": f"{cache_stats['hit_rate']:.2%}"
        }
    }

@app.get("/health")
async def health_check():
    # Import here to avoid circular imports
    from app.services.query_engine import query_engine
    
    health_status = await query_engine.health_check()
    return health_status

# Cache management endpoints
@app.get("/cache/stats")
async def get_cache_stats():
    """Get detailed cache statistics"""
    from app.services.query_engine import query_engine
    
    stats = await query_engine.get_processing_stats()
    return stats

@app.post("/cache/clear")
async def clear_cache(cache_type: str = "all"):
    """Clear cache by type (all, documents, embeddings, queries)"""
    if cache_type == "all":
        await cache_service.clear_cache()
    elif cache_type == "documents":
        await cache_service.clear_cache(cache_service.document_cache_prefix)
    elif cache_type == "embeddings":
        await cache_service.clear_cache(cache_service.embedding_cache_prefix)
    elif cache_type == "queries":
        await cache_service.clear_cache(cache_service.query_cache_prefix)
    else:
        return {"error": "Invalid cache type. Use: all, documents, embeddings, or queries"}
    
    return {"message": f"Cache cleared: {cache_type}"}

@app.post("/cache/cleanup")
async def manual_cache_cleanup():
    """Manually trigger cache cleanup"""
    await cache_service.cleanup_expired()
    stats = await cache_service.get_cache_stats()
    return {
        "message": "Cache cleanup completed",
        "remaining_entries": stats['total_entries']
    }

# System warm-up endpoint
@app.post("/system/warmup")
async def warmup_system(document_urls: list = None, common_questions: list = None):
    """Warm up the system with common documents and questions"""
    from app.services.query_engine import query_engine
    
    if not document_urls:
        document_urls = []
    if not common_questions:
        common_questions = []
    
    try:
        await query_engine.warm_up_system(document_urls, common_questions)
        return {
            "message": "System warm-up completed",
            "documents_processed": len(document_urls),
            "questions_precomputed": len(common_questions)
        }
    except Exception as e:
        return {"error": f"Warm-up failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
