"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     DAWN - BATCH KNOWLEDGE INJECTION                          ║
║                        "The Matrix Upload Method"                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  This script downloads and ingests entire knowledge datasets into DAWN's      ║
║  long-term ChromaDB memory using GPU-accelerated batch processing.            ║
║                                                                               ║
║  RTX 3050 6GB: ~32 batch size optimal | ~5,000-8,000 articles/hour            ║
║  RTX 3060 12GB: ~64 batch size optimal | ~10,000 articles/hour                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
#                               CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Memory storage path (use your HDD/SSD with lots of space)
MEMORY_PATH = "D:/DAWN/memory/long_term"

# Batch size - Adjust based on your GPU VRAM
# RTX 3050 6GB: Use 32 | RTX 3060 12GB: Use 64 | RTX 4090: Use 128
BATCH_SIZE = 32

# Maximum text length per chunk (characters)
MAX_CHUNK_SIZE = 1500

# Minimum text length to process (skip garbage)
MIN_TEXT_LENGTH = 100

# Available datasets to ingest
AVAILABLE_DATASETS = {
    "wikipedia": {
        "name": "wikitext",
        "config": "wikitext-103-v1",
        "split": "train",
        "text_field": "text",
        "description": "Wikipedia articles (~100MB, 1.8M sentences)"
    },
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "text_field": "text",
        "description": "Simple stories for reasoning (~500MB)"
    },
    "simple_wiki": {
        "name": "wikipedia",
        "config": "20220301.simple",
        "split": "train",
        "text_field": "text",
        "description": "Simple English Wikipedia (~200MB)"
    },
    "bookcorpus": {
        "name": "bookcorpus",
        "config": None,
        "split": "train",
        "text_field": "text",
        "description": "11,000 books (~5GB) - Large dataset"
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#                               UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Display startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         ██████╗  █████╗ ██╗    ██╗███╗   ██╗                     ║
    ║         ██╔══██╗██╔══██╗██║    ██║████╗  ██║                     ║
    ║         ██║  ██║███████║██║ █╗ ██║██╔██╗ ██║                     ║
    ║         ██║  ██║██╔══██║██║███╗██║██║╚██╗██║                     ║
    ║         ██████╔╝██║  ██║╚███╔███╔╝██║ ╚████║                     ║
    ║         ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝                     ║
    ║                                                                   ║
    ║              BATCH KNOWLEDGE INJECTION SYSTEM                     ║
    ║                  "The Matrix Upload Method"                       ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_gpu_info():
    """Get GPU information and status"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return True, f"{gpu_name} ({gpu_memory:.1f} GB)"
    return False, "No GPU detected - using CPU (slower)"


def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def clean_text(text):
    """Clean and normalize text for embedding"""
    if not text or not isinstance(text, str):
        return ""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove very short lines (often headers or noise)
    if len(text) < MIN_TEXT_LENGTH:
        return ""
    return text[:MAX_CHUNK_SIZE]


# ═══════════════════════════════════════════════════════════════════════════════
#                               MAIN INGESTION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DAWNKnowledgeInjector:
    def __init__(self, memory_path=MEMORY_PATH):
        self.memory_path = memory_path
        self.encoder = None
        self.client = None
        self.collection = None
        self.device = None
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": None
        }
    
    def initialize(self):
        """Initialize memory systems and encoder"""
        print("\n>>> INITIALIZING DAWN NEURAL SYSTEMS...")
        
        # Check GPU
        has_gpu, gpu_info = get_gpu_info()
        self.device = "cuda" if has_gpu else "cpu"
        print(f"    ├─ Device: {gpu_info}")
        
        # Create memory directory
        os.makedirs(self.memory_path, exist_ok=True)
        print(f"    ├─ Memory Path: {self.memory_path}")
        
        # Initialize ChromaDB
        print("    ├─ Mounting ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.memory_path)
        self.collection = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"description": "DAWN's long-term knowledge storage"}
        )
        existing_count = self.collection.count()
        print(f"    ├─ Existing memories: {existing_count:,}")
        
        # Load encoder on GPU
        print("    └─ Loading Neural Encoder (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        print("\n✓ SYSTEMS READY")
        return self
    
    def ingest_dataset(self, dataset_key="wikipedia", max_records=None):
        """
        Ingest a HuggingFace dataset into DAWN's memory
        
        Args:
            dataset_key: Key from AVAILABLE_DATASETS
            max_records: Limit number of records (None = all)
        """
        if dataset_key not in AVAILABLE_DATASETS:
            print(f"ERROR: Unknown dataset '{dataset_key}'")
            print(f"Available: {list(AVAILABLE_DATASETS.keys())}")
            return
        
        config = AVAILABLE_DATASETS[dataset_key]
        print(f"\n>>> LOADING DATASET: {config['description']}")
        
        # Load dataset from HuggingFace
        if config["config"]:
            dataset = load_dataset(config["name"], config["config"], split=config["split"])
        else:
            dataset = load_dataset(config["name"], split=config["split"])
        
        total_records = len(dataset)
        if max_records:
            total_records = min(max_records, total_records)
        
        print(f"    ├─ Total Records: {total_records:,}")
        print(f"    ├─ Batch Size: {BATCH_SIZE}")
        print(f"    └─ Text Field: {config['text_field']}")
        
        # Get current memory offset for unique IDs
        current_count = self.collection.count()
        
        # Start ingestion
        self.stats["start_time"] = time.time()
        self._batch_ingest(dataset, config["text_field"], total_records, 
                          dataset_key, current_count)
        
        # Print final stats
        elapsed = time.time() - self.stats["start_time"]
        print(f"\n╔═══════════════════════════════════════════════════════════════╗")
        print(f"║                    INGESTION COMPLETE                         ║")
        print(f"╠═══════════════════════════════════════════════════════════════╣")
        print(f"║  Processed: {self.stats['processed']:>10,} records                      ║")
        print(f"║  Skipped:   {self.stats['skipped']:>10,} (too short/empty)              ║")
        print(f"║  Errors:    {self.stats['errors']:>10,}                                 ║")
        print(f"║  Time:      {format_time(elapsed):>10}                                  ║")
        print(f"║  Speed:     {self.stats['processed']/elapsed*3600:>10,.0f} records/hour            ║")
        print(f"║  Total Memory: {self.collection.count():>7,} vectors                     ║")
        print(f"╚═══════════════════════════════════════════════════════════════╝")
    
    def _batch_ingest(self, dataset, text_field, total_records, source_name, id_offset):
        """Process dataset in GPU-optimized batches"""
        batch_docs = []
        batch_metas = []
        batch_ids = []
        
        progress_bar = tqdm(
            enumerate(dataset),
            total=total_records,
            desc="Ingesting Knowledge",
            unit="docs",
            ncols=100
        )
        
        for i, record in progress_bar:
            if i >= total_records:
                break
            
            # Extract and clean text
            raw_text = record.get(text_field, "")
            text = clean_text(raw_text)
            
            if not text:
                self.stats["skipped"] += 1
                continue
            
            # Add to batch
            batch_docs.append(text)
            batch_metas.append({
                "source": source_name,
                "timestamp": datetime.now().isoformat(),
                "chunk_index": i
            })
            batch_ids.append(f"{source_name}_{id_offset + i}")
            
            # Process batch when full
            if len(batch_docs) >= BATCH_SIZE:
                self._process_batch(batch_docs, batch_metas, batch_ids)
                batch_docs = []
                batch_metas = []
                batch_ids = []
                
                # Update progress bar
                progress_bar.set_postfix({
                    "processed": self.stats["processed"],
                    "memory": self.collection.count()
                })
        
        # Process remaining items
        if batch_docs:
            self._process_batch(batch_docs, batch_metas, batch_ids)
    
    def _process_batch(self, docs, metas, ids):
        """Process a single batch with GPU acceleration"""
        try:
            # Generate embeddings on GPU (FAST)
            embeddings = self.encoder.encode(
                docs,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Store in ChromaDB
            self.collection.add(
                documents=docs,
                embeddings=embeddings.tolist(),
                metadatas=metas,
                ids=ids
            )
            
            self.stats["processed"] += len(docs)
            
        except Exception as e:
            self.stats["errors"] += len(docs)
            print(f"\n⚠ Batch error: {str(e)[:50]}...")
    
    def query_memory(self, question, n_results=5):
        """Query DAWN's memory for relevant knowledge"""
        print(f"\n>>> QUERYING MEMORY: '{question}'")
        
        # Encode question
        query_embedding = self.encoder.encode([question])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        print(f"\n┌─ TOP {n_results} MEMORIES ─────────────────────────────────┐")
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"│ [{i+1}] Source: {meta.get('source', 'unknown')}")
            print(f"│     {doc[:100]}...")
            print("│")
        print("└──────────────────────────────────────────────────────────┘")
        
        return results
    
    def get_stats(self):
        """Get current memory statistics"""
        count = self.collection.count()
        return {
            "total_memories": count,
            "memory_path": self.memory_path,
            "device": self.device
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                               MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner()
    
    # Initialize DAWN
    dawn = DAWNKnowledgeInjector()
    dawn.initialize()
    
    # Show available datasets
    print("\n>>> AVAILABLE KNOWLEDGE SOURCES:")
    for key, config in AVAILABLE_DATASETS.items():
        print(f"    • {key}: {config['description']}")
    
    # Default: Ingest Wikipedia (fastest, most reliable)
    # For testing, limit to 10,000 records first
    print("\n" + "="*70)
    print("Starting with Wikipedia dataset (testing with 10,000 records)...")
    print("="*70)
    
    dawn.ingest_dataset("wikipedia", max_records=10000)
    
    # Test query
    print("\n>>> TESTING MEMORY RECALL...")
    dawn.query_memory("What is the theory of relativity?")
    dawn.query_memory("How do computers work?")
    
    print("\n✓ DAWN IS NOW EDUCATED. Memory saved to:", MEMORY_PATH)


if __name__ == "__main__":
    main()
