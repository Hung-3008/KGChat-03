import os
import sys
from qdrant_client import QdrantClient
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def check_status():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=10)
    collection_name = "kg_lv2_nodes"

    print(f"Checking status for '{collection_name}' at {url}...")
    
    try:
        info = client.get_collection(collection_name)
        
        print("\n--- Collection Info ---")
        print(f"Status: {info.status}")
        print(f"Vectors Count: {info.vectors_count}")
        print(f"Indexed Vectors Count: {info.indexed_vectors_count}")
        print(f"Points Count: {info.points_count}")
        
        print("\n--- Configuration ---")
        # Dump config to string to see nested objects clearly
        config_dict = info.config.dict()
        
        print(f"Full Config: {json.dumps(config_dict, indent=2)}")
        
        # Check HNSW
        hnsw_config = config_dict.get('params', {}).get('hnsw_config', {})
        print(f"HNSW Config: {json.dumps(hnsw_config, indent=2)}")

        # Check Optimizer
        optimizer_config = config_dict.get('optimizer_config', {})
        print(f"Optimizer Config: {json.dumps(optimizer_config, indent=2)}")

        if info.status != 'green':
            print("\n⚠️  WARNING: Collection status is NOT green. Optimization might be still running.")
        else:
            print("\n✅ Collection status is GREEN.")

    except Exception as e:
        print(f"❌ Failed to get collection info: {e}")

if __name__ == "__main__":
    check_status()
