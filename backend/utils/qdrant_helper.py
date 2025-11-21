import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
import logging

logger = logging.getLogger("qdrant_helper")

class QdrantHelper:
    def __init__(self):
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY", None)
        
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info("Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, collection_name: str, vector_size: int = 768):
        """Creates collection if it doesn't exist."""
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logger.info(f"Created collection '{collection_name}'")
        else:
            logger.info(f"Collection '{collection_name}' already exists")

    def clear_collection(self, collection_name: str):
        """Deletes and recreates the collection."""
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
        # Re-create is handled by create_collection called subsequently or explicitly here if needed.
        # For 'clear', we usually just delete. The caller should re-create.

    def insert_points(self, collection_name: str, points: List[Dict]):
        """
        Batch insert points.
        Expected point dict: {'id': str (uuid), 'vector': List[float], 'payload': Dict}
        """
        if not points:
            return

        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point['id'],
                    vector=point['vector'],
                    payload=point.get('payload', {})
                )
                for point in points
            ]
        )
