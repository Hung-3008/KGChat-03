import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
import logging
import time
import httpx
from qdrant_client.http.exceptions import ResponseHandlingException

logger = logging.getLogger("qdrant_helper")

class QdrantHelper:
    def __init__(self):
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY", None)
        
        try:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=600)
            logger.info("Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, collection_name: str, vector_size: int = 768):
        """Creates collection if it doesn't exist."""
        exists = False
        try:
            exists = self.client.collection_exists(collection_name)
        except Exception:
            # If 404 or other error, assume it doesn't exist
            exists = False
            
        if not exists:
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
                logger.info(f"Created collection '{collection_name}'")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Collection '{collection_name}' already exists (caught during creation)")
                else:
                    logger.error(f"Failed to create collection: {e}")
                    raise
        else:
            logger.info(f"Collection '{collection_name}' already exists")

    def clear_collection(self, collection_name: str):
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
        except Exception as e:
            # If collection doesn't exist, Qdrant might raise an error or return false.
            # We log it but don't fail, as the goal is to ensure it's gone.
            logger.info(f"Attempted to delete collection '{collection_name}', result: {e}")
        
        # Re-create is handled by create_collection called subsequently or explicitly here if needed.
        # For 'clear', we usually just delete. The caller should re-create.

    def insert_points(self, collection_name: str, points: List[Dict]):
        """
        Batch insert points.
        Expected point dict: {'id': str (uuid), 'vector': List[float], 'payload': Dict}
        """
        if not points:
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
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
                return  # Success
            except (ResponseHandlingException, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Insert failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Insert failed after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error during insert: {e}")
                raise

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5, score_threshold: float = 0.65) -> List[Dict]:
        """
        Search for similar vectors in the collection.
        Returns a list of point IDs that have a similarity score > score_threshold.
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True 
            )
            
            # Filter by score and extract IDs and names
            filtered_results = []
            for point in results:
                if point.score > score_threshold:
                    payload = point.payload or {}
                    filtered_results.append({
                        "id": str(point.id),
                        "name": payload.get("name", "Unknown")
                    })
            
            logger.info(f"Found {len(results)} results, {len(filtered_results)} passed threshold {score_threshold}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
            
    def get_collection_count(self, collection_name: str) -> int:
        """Returns the number of points in the collection."""
        try:
            count_result = self.client.count(collection_name=collection_name)
            return count_result.count
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0

