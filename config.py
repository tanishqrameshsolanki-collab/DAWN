"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     DAWN - CONFIGURATION FILE                                 ║
║              Adjust these settings based on your hardware                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
#                               PATHS
# ═══════════════════════════════════════════════════════════════════════════════

# Base directory for all DAWN data
BASE_DIR = "D:/DAWN"

# Long-term memory (ChromaDB vectors)
MEMORY_DIR = os.path.join(BASE_DIR, "memory/long_term")

# Working memory (temporary/session data)
WORKING_MEMORY_DIR = os.path.join(BASE_DIR, "memory/working")

# Logs directory
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Downloaded datasets cache
CACHE_DIR = os.path.join(BASE_DIR, "cache")


# ═══════════════════════════════════════════════════════════════════════════════
#                           GPU / PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

# Batch sizes optimized by GPU VRAM
GPU_BATCH_SIZES = {
    4: 16,    # 4GB VRAM (GTX 1650, etc.)
    6: 32,    # 6GB VRAM (RTX 3050, GTX 1660)
    8: 48,    # 8GB VRAM (RTX 3070, RTX 4060)
    12: 64,   # 12GB VRAM (RTX 3060, RTX 4070)
    16: 96,   # 16GB VRAM (RTX 4080)
    24: 128,  # 24GB VRAM (RTX 4090, A5000)
}

# Default batch size for your RTX 3050 6GB
DEFAULT_BATCH_SIZE = 32

# Embedding model (smaller = faster, larger = better quality)
EMBEDDING_MODELS = {
    "fast": "all-MiniLM-L6-v2",           # 384 dims, fastest
    "balanced": "all-mpnet-base-v2",       # 768 dims, good balance
    "quality": "all-roberta-large-v1",     # 1024 dims, best quality
}

DEFAULT_EMBEDDING_MODEL = "fast"


# ═══════════════════════════════════════════════════════════════════════════════
#                           TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum characters per text chunk
MAX_CHUNK_SIZE = 1500

# Minimum text length to process (skip garbage)
MIN_TEXT_LENGTH = 100

# Overlap between chunks for context continuity
CHUNK_OVERLAP = 100


# ═══════════════════════════════════════════════════════════════════════════════
#                           MEMORY SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Number of memories to retrieve per query
DEFAULT_RETRIEVAL_COUNT = 5

# Similarity threshold for memory recall (0.0 - 1.0)
SIMILARITY_THRESHOLD = 0.3


# ═══════════════════════════════════════════════════════════════════════════════
#                           DATASET SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_SOURCES = {
    # Core Knowledge (Recommended for first run)
    "wikipedia": {
        "name": "wikitext",
        "config": "wikitext-103-v1",
        "split": "train",
        "text_field": "text",
        "size_mb": 100,
        "priority": 1,
        "description": "Wikipedia articles - general knowledge foundation"
    },
    
    # Simple Language (Good for reasoning)
    "simple_wiki": {
        "name": "wikipedia",
        "config": "20220301.simple",
        "split": "train", 
        "text_field": "text",
        "size_mb": 200,
        "priority": 2,
        "description": "Simple English Wikipedia - clearer explanations"
    },
    
    # Stories (Good for understanding narrative/context)
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "text_field": "text",
        "size_mb": 500,
        "priority": 3,
        "description": "Simple stories - helps with reasoning chains"
    },
    
    # Large-scale (Only if you have time/space)
    "bookcorpus": {
        "name": "bookcorpus",
        "config": None,
        "split": "train",
        "text_field": "text",
        "size_mb": 5000,
        "priority": 10,
        "description": "11,000 books - massive knowledge (takes hours)"
    },
    
    # Scientific (Specialized knowledge)
    "pubmed": {
        "name": "ccdv/pubmed-summarization",
        "config": None,
        "split": "train",
        "text_field": "article",
        "size_mb": 2000,
        "priority": 5,
        "description": "Medical/scientific papers"
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                           LIVE LEARNING (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

# Multithreaded crawler settings (for "Hive Mind" mode)
CRAWLER_THREADS = 10  # Number of parallel browser instances
CRAWLER_TIMEOUT = 10  # Seconds to wait per page
CRAWLER_RATE_LIMIT = 0.5  # Seconds between requests per thread

# YouTube learning settings
YOUTUBE_API_KEY = None  # Add your API key for video transcripts
MAX_VIDEO_LENGTH = 1800  # 30 minutes max (in seconds)
