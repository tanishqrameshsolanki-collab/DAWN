"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     DAWN - THE MATRIX UPLOAD                                  ║
║              Automatic Knowledge Injection from Cloud Datasets                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  This script downloads Wikipedia/Books from HuggingFace and injects           ║
║  them directly into DAWN's long-term memory using GPU acceleration.           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
#                               CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Where the AI's brain will live on your Hard Drive
MEMORY_PATH = "D:/DAWN/memory/long_term"

# Batch size: Higher = Faster, but uses more VRAM
# RTX 3050 6GB: Use 32 | RTX 3060 12GB: Use 64 | RTX 4090: Use 128
BATCH_SIZE = 32  # Optimized for your RTX 3050 6GB

# Minimum text length to process (skip garbage/headers)
MIN_TEXT_LENGTH = 150

# Maximum characters per chunk
MAX_CHUNK_SIZE = 2000


# ═══════════════════════════════════════════════════════════════════════════════
#                               MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_wikipedia():
    """Download and ingest Wikipedia into DAWN's memory"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         ██████╗  █████╗ ██╗    ██╗███╗   ██╗                     ║
    ║         ██╔══██╗██╔══██╗██║    ██║████╗  ██║                     ║
    ║         ██║  ██║███████║██║ █╗ ██║██╔██╗ ██║                     ║
    ║         ██║  ██║██╔══██║██║███╗██║██║╚██╗██║                     ║
    ║         ██████╔╝██║  ██║╚███╔███╔╝██║ ╚████║                     ║
    ║         ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═══╝                     ║
    ║                                                                   ║
    ║                   THE MATRIX UPLOAD                               ║
    ║              "Downloading Knowledge to Brain..."                  ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Initialize GPU
    # ─────────────────────────────────────────────────────────────────────
    print(">>> 1. INITIALIZING GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"    ✓ GPU DETECTED: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"    ✓ CUDA Version: {torch.version.cuda}")
    else:
        print("    ⚠ NO GPU DETECTED - Running on CPU (slower)")
    
    # Load the neural encoder
    print("    └─ Loading Neural Encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Connect to Memory Core
    # ─────────────────────────────────────────────────────────────────────
    print("\n>>> 2. CONNECTING TO MEMORY CORE...")
    os.makedirs(MEMORY_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=MEMORY_PATH)
    collection = client.get_or_create_collection(name="long_term")
    existing_count = collection.count()
    print(f"    ✓ Memory Path: {MEMORY_PATH}")
    print(f"    ✓ Existing Memories: {existing_count:,}")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Download Dataset
    # ─────────────────────────────────────────────────────────────────────
    print("\n>>> 3. DOWNLOADING DATASET (WikiText-103)...")
    print("    └─ This is a high-quality subset of Wikipedia (~100MB)")
    
    # WikiText-103-v1 is a cleaned, high-quality subset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    total_articles = len(dataset)
    print(f"    ✓ Downloaded: {total_articles:,} articles")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Batch Ingestion
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n>>> 4. STARTING BATCH INGESTION...")
    print(f"    ├─ Batch Size: {BATCH_SIZE}")
    print(f"    ├─ Min Text Length: {MIN_TEXT_LENGTH} chars")
    print(f"    └─ Max Chunk Size: {MAX_CHUNK_SIZE} chars")
    print()
    
    batch_docs = []
    batch_metas = []
    batch_ids = []
    processed = 0
    skipped = 0
    
    # Progress bar
    progress = tqdm(
        enumerate(dataset),
        total=total_articles,
        desc="Injecting Knowledge",
        unit="docs",
        ncols=100
    )
    
    for i, record in progress:
        text = record['text'].strip()
        
        # Skip empty lines or tiny headers
        if len(text) < MIN_TEXT_LENGTH:
            skipped += 1
            continue
        
        # Prepare the data (slice to keep memories focused)
        batch_docs.append(text[:MAX_CHUNK_SIZE])
        batch_metas.append({
            "source": "WikiText-103",
            "topic": f"Article_{i}",
            "char_count": len(text)
        })
        batch_ids.append(f"wiki_{existing_count + i}")
        
        # When batch is full, BLAST it to the GPU
        if len(batch_docs) >= BATCH_SIZE:
            # Vectorize on GPU (FAST!)
            embeddings = encoder.encode(batch_docs, show_progress_bar=False)
            
            # Save to ChromaDB
            collection.add(
                documents=batch_docs,
                embeddings=embeddings.tolist(),
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            processed += len(batch_docs)
            
            # Reset batch
            batch_docs = []
            batch_metas = []
            batch_ids = []
            
            # Update progress bar
            progress.set_postfix({
                "stored": processed,
                "skipped": skipped
            })
    
    # Process remaining items
    if batch_docs:
        embeddings = encoder.encode(batch_docs, show_progress_bar=False)
        collection.add(
            documents=batch_docs,
            embeddings=embeddings.tolist(),
            metadatas=batch_metas,
            ids=batch_ids
        )
        processed += len(batch_docs)
    
    # ─────────────────────────────────────────────────────────────────────
    # COMPLETE
    # ─────────────────────────────────────────────────────────────────────
    final_count = collection.count()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    ✓ UPLOAD COMPLETE                              ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Processed:    {processed:>10,} articles                          ║
    ║  Skipped:      {skipped:>10,} (too short)                         ║
    ║  Total Memory: {final_count:>10,} vectors                         ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  DAWN now has access to encyclopedic knowledge!                   ║
    ║  Memory saved to: {MEMORY_PATH:<43} ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    return collection


def test_memory():
    """Test DAWN's memory recall"""
    print("\n>>> TESTING MEMORY RECALL...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    client = chromadb.PersistentClient(path=MEMORY_PATH)
    collection = client.get_or_create_collection(name="long_term")
    
    test_queries = [
        "What is the theory of relativity?",
        "How do computers work?",
        "What is photosynthesis?",
        "Tell me about World War II",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        query_embedding = encoder.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        print("   Top Memories:")
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"   [{i+1}] {doc[:100]}...")


if __name__ == "__main__":
    ingest_wikipedia()
    test_memory()
