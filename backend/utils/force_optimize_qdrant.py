
import os
import sys
import time
from qdrant_client import QdrantClient, models

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def force_optimize():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=600)
    collection_name = "kg_lv2_nodes"
    
    print(f"Connecting to Qdrant at {url}...")
    if not client.collection_exists(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist!")
        return

    # 1. Force merge segments
    print(f"Initiating optimization for '{collection_name}'...")
    print("Target: 2 segments (optimal for search speed vs update capability)")
    
    try:
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                default_segment_number=2,
                indexing_threshold=10000, 
                memmap_threshold=20000 
            )
        )
        print("Optimization config updated.")
        
        # Is there a way to trigger explicit optimization? 
        # Usually update_collection triggers it if config changes. 
        # But to be sure, we can't manually 'trigger' merge via client easily without config change.
        # Ideally, changing default_segment_number to 2 (if it wasn't) will trigger it.
        
    except Exception as e:
        print(f"⚠️ Note: Config update might have failed or is redundant: {e}")
        print("Proceeding to monitor status...")


    # 2. Monitor loop
    print("Waiting for optimization to complete...")
    start_time = time.time()
    
    while True:
        try:
            info = client.get_collection(collection_name)
            status = info.status
            segments = info.segments_count
            
            elapsed = time.time() - start_time
            print(f"[{elapsed:.0f}s] Status: {status} | Segments: {segments}")
            
            if status == models.CollectionStatus.GREEN and segments <= 2:
                print("✅ Optimization Complete!")
                break
                
            time.sleep(5)
            
        except Exception as e:
            print(f"Error monitoring status: {e}")
            time.sleep(5)

if __name__ == "__main__":
    force_optimize()
