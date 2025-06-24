import os
import logging
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import ScoredPoint, PointStruct
import uuid
from dotenv import load_dotenv

from .base_client import VectorDatabaseClient

logger = logging.getLogger(__name__)


class QdrantVectorClient(VectorDatabaseClient):
    """
    Qdrant implementation of VectorDatabaseClient.

    Client for interacting with Qdrant vector database for knowledge graph nodes.
    Handles vector storage, retrieval, and similarity search operations.
    """

    def __init__(
        self,
        host: str = None,
        port: int = 6333,
        api_key: str = None,
        url: str = None,
        vector_size: int = 1024
    ):
        """
        Initialize Qdrant vector client.

        Args:
            host: Qdrant server host (default: localhost)
            port: Qdrant server port (default: 6333)
            api_key: API key for Qdrant Cloud
            url: URL for Qdrant Cloud
            vector_size: Dimension of vector embeddings (default: 1024)
        """
        super().__init__()

        # Load environment variables if not provided
        load_dotenv()

        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.url = url or os.getenv("QDRANT_URL")
        self.vector_size = vector_size

        # Initialize the Qdrant client connection
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client with appropriate connection type."""
        try:
            if self.url:
                # Cloud connection using URL and API key
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
                logger.info(
                    f"Initialized Qdrant client with cloud URL: {self.url}")
            else:
                # Local connection using host and port
                self.client = QdrantClient(host=self.host, port=self.port)
                logger.info(
                    f"Initialized Qdrant client with local host: {self.host}:{self.port}")

            # Set connection properties
            self._connection = self.client
            self._is_connected = True
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            self._is_connected = False
            raise

    async def connect(self) -> bool:
        """
        Connect to Qdrant database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._initialize_client()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            return False

    async def close(self) -> None:
        """Close Qdrant connection."""
        if self.client:
            try:
                self.client.close()
                self._is_connected = False
                logger.info("Qdrant connection closed")
            except Exception as e:
                logger.warning(f"Error closing Qdrant connection: {str(e)}")

    async def verify_connectivity(self) -> bool:
        """
        Verify Qdrant connection is working.

        Returns:
            True if connected and working, False otherwise
        """
        try:
            # Test connection by getting collections list
            collections = self.client.get_collections()
            logger.info("Qdrant connectivity verified")
            return True
        except Exception as e:
            logger.error(f"Qdrant connectivity check failed: {str(e)}")
            return False

    async def setup_schema(self) -> bool:
        """
        Set up collections for Qdrant.

        Creates level1_nodes and level2_nodes collections for knowledge graph.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            collection_configs = [
                {
                    "name": "level1_nodes",
                    "description": "Collection for Level 1 knowledge graph nodes (general concepts)"
                },
                {
                    "name": "level2_nodes",
                    "description": "Collection for Level 2 knowledge graph nodes (specific details)"
                }
            ]

            return await self.create_collections(collection_configs)
        except Exception as e:
            logger.error(f"Error setting up Qdrant schema: {str(e)}")
            return False

    async def clear_database(self) -> bool:
        """
        Delete all collections in Qdrant database.

        WARNING: This deletes all vector data!

        Returns:
            True if clearing successful, False otherwise
        """
        try:
            collections = self.client.get_collections()

            # Delete each collection
            for collection in collections.collections:
                self.client.delete_collection(collection.name)
                logger.info(f"Deleted collection: {collection.name}")

            logger.info(
                "Successfully cleared all collections from Qdrant database")
            return True
        except Exception as e:
            logger.error(f"Error clearing Qdrant database: {str(e)}")
            return False

    async def import_data(self, data: List[Dict[str, Any]], collection_name: str = None, **kwargs) -> bool:
        """
        Import vector data into Qdrant.

        Args:
            data: List of vector data dictionaries
            collection_name: Target collection name
            **kwargs: Additional parameters

        Returns:
            True if import successful, False otherwise
        """
        if not collection_name:
            logger.error("Collection name is required for vector data import")
            return False

        imported_count = await self.store_vectors(data, collection_name, **kwargs)
        return imported_count > 0

    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform vector search in Qdrant.

        Args:
            query_params: Query parameters including vector, collection_name, limit, score_threshold

        Returns:
            List of search results with scores
        """
        query_vector = query_params.get('vector', [])
        collection_name = query_params.get('collection_name', 'level1_nodes')
        limit = query_params.get('limit', 10)
        score_threshold = query_params.get('score_threshold', 0.5)

        if not query_vector:
            logger.error("Query vector is required for vector search")
            return []

        return await self.search_vectors(query_vector, collection_name,
                                         limit=limit, score_threshold=score_threshold)

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Qdrant collections.

        Returns:
            Dictionary containing collection statistics
        """
        try:
            collections = self.client.get_collections()
            stats = {
                "total_collections": len(collections.collections),
                "collections": {}
            }

            total_points = 0
            # Get info for each collection
            for collection in collections.collections:
                try:
                    collection_info = self.client.get_collection(
                        collection.name)
                    points_count = collection_info.points_count or 0
                    stats["collections"][collection.name] = {
                        "points_count": points_count,
                        "status": collection_info.status,
                        "vector_size": collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 'unknown'
                    }
                    total_points += points_count
                except Exception as e:
                    logger.warning(
                        f"Error getting info for collection {collection.name}: {str(e)}")
                    stats["collections"][collection.name] = {"error": str(e)}

            stats["total_points"] = total_points

            logger.info(f"Qdrant statistics retrieved: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting Qdrant statistics: {str(e)}")
            return {}

    # Vector database specific methods
    async def create_collections(self, collection_configs: List[Dict[str, Any]]) -> bool:
        """
        Create collections for level 1 and level 2 nodes if they don't exist.

        Args:
            collection_configs: List of collection configuration dictionaries

        Returns:
            True if creation successful, False otherwise
        """
        try:
            for config in collection_configs:
                collection_name = config.get("name")

                if not collection_name:
                    logger.warning("Collection name is required in config")
                    continue

                try:
                    # Check if collection already exists
                    self.client.get_collection(collection_name)
                    logger.info(f"Collection {collection_name} already exists")
                except Exception:
                    # Create new collection if it doesn't exist
                    logger.info(f"Creating collection {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    logger.info(f"Created collection {collection_name}")

            return True
        except Exception as e:
            logger.error(f"Error creating collections: {str(e)}")
            return False

    async def store_vectors(self, vectors: List[Dict[str, Any]], collection_name: str, **kwargs) -> int:
        """
        Store vectors in Qdrant collection.

        Args:
            vectors: List of vector data dictionaries with embeddings and metadata
            collection_name: Target collection name
            **kwargs: Additional parameters

        Returns:
            Number of vectors successfully stored
        """
        try:
            logger.info(
                f"Received {len(vectors)} vectors for storing in collection {collection_name}")

            valid_points = []
            # Process each vector data
            for vector_data in vectors:
                # Generate unique point ID
                point_id = str(uuid.uuid4())

                # Extract vector embedding
                vector = vector_data.get("vector_embedding", [])

                if not vector:
                    logger.warning(
                        f"Vector {vector_data.get('entity_id', 'unknown')} has no vector embedding, skipping")
                    continue

                # Create payload with metadata (excluding unnecessary fields)
                payload = {
                    "original_entity_id": vector_data.get("entity_id", point_id),
                    "knowledge_level": vector_data.get("knowledge_level", 1),
                    **{k: v for k, v in vector_data.items()
                       if k not in ["entity_id", "vector_embedding", "name", "type", "description",
                                    "knowledge_level", "entity_name", "entity_type"]}
                }

                # Create point structure for Qdrant
                valid_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            logger.info(f"Found {len(valid_points)} valid vectors to store")

            # Upsert vectors to collection
            if valid_points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=valid_points
                )

                logger.info(
                    f"Stored {len(valid_points)} vectors in {collection_name}")
                return len(valid_points)

            logger.warning("No valid vectors to store")
            return 0

        except Exception as e:
            logger.error(
                f"Error storing vectors in collection {collection_name}: {str(e)}")
            return 0

    async def search_vectors(self, query_vector: List[float], collection_name: str,
                             limit: int = 10, score_threshold: float = 0.5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.

        Args:
            query_vector: Vector to search with
            collection_name: Collection to search in
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            **kwargs: Additional search parameters

        Returns:
            List of similar vectors with scores and metadata
        """
        try:
            # Perform vector similarity search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=kwargs.get('with_vectors', False)
            )

            # Format results
            results = []
            for point in search_result:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload or {}
                }

                # Include vector if requested
                if kwargs.get('with_vectors', False) and point.vector:
                    result["vector"] = point.vector

                results.append(result)

            logger.info(
                f"Found {len(results)} similar vectors in {collection_name}")
            return results

        except Exception as e:
            logger.error(
                f"Error searching vectors in collection {collection_name}: {str(e)}")
            return []

    async def retrieve_by_ids(self, ids: List[str], collection_name: str,
                              with_vectors: bool = False, with_payload: bool = True,
                              limit: int = 10, score_threshold: float = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve vectors by their IDs from Qdrant.

        Args:
            ids: List of point IDs to retrieve
            collection_name: Collection to retrieve from
            with_vectors: Whether to include vector data
            with_payload: Whether to include metadata
            limit: Maximum number of results
            score_threshold: Optional score threshold filter
            **kwargs: Additional parameters

        Returns:
            List of retrieved vectors with metadata
        """
        try:
            # Retrieve points by IDs
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_vectors=with_vectors,
                with_payload=with_payload
            )

            # Format results
            results = []
            for point in points[:limit]:
                result = {
                    "id": point.id,
                    "payload": point.payload if with_payload else None,
                    "score": 1.0  # Default score for direct retrieval
                }

                # Include vector if requested
                if with_vectors and point.vector:
                    result["vector"] = point.vector

                # Apply score threshold filter if specified
                if score_threshold is None or result["score"] >= score_threshold:
                    results.append(result)

            logger.info(
                f"Retrieved {len(results)} points by ID from collection {collection_name}")
            return results

        except Exception as e:
            logger.error(
                f"Error retrieving points by ID from collection {collection_name}: {str(e)}")
            return []
