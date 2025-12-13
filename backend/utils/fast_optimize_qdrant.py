
import os
import logging
from qdrant_client import QdrantClient, models

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qdrant_optimizer")

def optimize_collection():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)
    collection_name = "kg_lv2_nodes"
    
    logger.info(f"Connecting to Qdrant at {url}...")
    client = QdrantClient(url=url, api_key=api_key, timeout=60)

    # 1. Update Quantization Config (Scalar Quantization - Int8)
    logger.info(f"Enabling Scalar Quantization (Int8) for '{collection_name}'...")
    try:
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                # default_segments_number=2, # Removed as it caused extra_forbidden error
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )
        )
        logger.info("Quantization configuration updated.")
    except Exception as e:
        logger.error(f"Failed to update quantization config: {e}")

    # 2. Update HNSW Index Config
    logger.info(f"Updating HNSW config for '{collection_name}'...")
    try:
        client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,             # Links per node (16-64)
                ef_construct=100, # Construction accuracy
                # on_disk=True    # Uncomment if RAM is critically low
            )
        )
        logger.info("HNSW configuration updated.")
    except Exception as e:
        logger.error(f"Failed to update HNSW config: {e}")

    logger.info("Optimization requests sent. internal optimization processes will run in background.")

if __name__ == "__main__":
    optimize_collection()
