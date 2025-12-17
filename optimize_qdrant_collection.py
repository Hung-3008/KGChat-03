import sys
import os
import logging
from qdrant_client import models

# Import QdrantHelper to reuse connection logic
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from backend.utils.qdrant_helper import QdrantHelper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimize_qdrant")

def optimize_collection(collection_name: str):
    logger.info(f"Connecting to Qdrant to optimize collection: {collection_name}")
    helper = QdrantHelper()
    client = helper.client
    
    # 1. Update Collection for Scalar Quantization (Int8)
    logger.info("Enabling Scalar Quantization (Int8)...")
    try:
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                # default_segments_number=2, # Removed as it caused validation error
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True # Keep quantized vectors in RAM for speed
                )
            )
        )
        logger.info("Scalar Quantization enabled.")
    except Exception as e:
        logger.error(f"Failed to enable quantization: {e}")

    # 2. Optimize HNSW Index Config
    logger.info("Updating HNSW Index Config...")
    try:
        client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,             # 16-64 links per node
                ef_construct=100, # Construction accuracy
                # on_disk=True    # Uncomment if RAM is still an issue, but slower
            )
        )
        logger.info("HNSW Index Config updated.")
    except Exception as e:
        logger.error(f"Failed to update HNSW config: {e}")

    logger.info("Optimization process initiated. Qdrant will apply changes in background.")

if __name__ == "__main__":
    COLLECTION_NAME = "kg_lv2_nodes"
    optimize_collection(COLLECTION_NAME)
