import os
import sys
import time
from qdrant_client import QdrantClient, models

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def optimize_collection():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=600)
    collection_name = "kg_lv2_nodes"

    print(f"Connecting to Qdrant at {url}...")
    
    if not client.collection_exists(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist!")
        return

    print(f"Optimizing collection '{collection_name}' for large-scale search...")
    print("Strategy: Binary Quantization + Rescoring + On-Disk HNSW")

    try:
        # Update collection configuration
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                default_segment_number=2, # Optimizing for search speed
                indexing_threshold=10000,
                memmap_threshold=20000, # Force storage on disk for vectors
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True,  # Keep compressed index in RAM for ultra-fast search
                ),
            ),
            hnsw_config=models.HnswConfigDiff(
                on_disk=False, # Keep HNSW graph in RAM for speed (Crucial for performance)
                m=16,
                ef_construct=100,
            )
        )
        print("✅ Optimization request sent successfully!")
        print("The optimization process (quantization & re-indexing) is running in the background.")
        print("You can monitor the Qdrant logs to see the progress.")
        
    except Exception as e:
        print(f"❌ Failed to update collection: {e}")

if __name__ == "__main__":
    optimize_collection()
