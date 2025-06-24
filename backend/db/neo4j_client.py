import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
import os
from dotenv import load_dotenv
import uuid

from .base_client import GraphDatabaseClient

logger = logging.getLogger(__name__)


class Neo4jClient(GraphDatabaseClient):
    """
    Neo4j implementation of GraphDatabaseClient.

    Handles connection and operations with Neo4j database for knowledge graph.
    Supports both synchronous and asynchronous operations for graph data management.
    """

    def __init__(self, uri: str = None,
                 username: str = None,
                 password: str = None,
                 database: str = "neo4j",
                 use_async: bool = True
                 ):
        """
        Initialize Neo4j client with connection parameters.

        Args:
            uri: Neo4j connection URI (default: bolt://localhost:7687)
            username: Neo4j username (default: neo4j)
            password: Neo4j password
            database: Target database name (default: neo4j)
            use_async: Whether to use async operations (default: True)
        """
        super().__init__()

        # Load environment variables if parameters not provided
        if uri is None or username is None or password is None:
            load_dotenv()
            uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            username = username or os.getenv("NEO4J_USERNAME", "neo4j")
            password = password or os.getenv("NEO4J_PASSWORD", "password")

        # Set instance attributes
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.use_async = use_async
        self._driver = None

        # Initialize connection asynchronously
        asyncio.create_task(self.connect())

    async def connect(self) -> bool:
        """
        Connect to Neo4j database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create appropriate driver based on async preference
            if self.use_async:
                self._driver = AsyncGraphDatabase.driver(
                    self.uri, auth=(self.username, self.password))
            else:
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.username, self.password))

            self._is_connected = True
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self._is_connected = False
            return False

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            if self.use_async:
                await self._driver.close()
            else:
                self._driver.close()
            self._is_connected = False
            logger.info("Neo4j connection closed")

    def close_sync(self):
        """Synchronous version of close method for non-async usage."""
        if self._driver and not self.use_async:
            self._driver.close()
            self._is_connected = False
            logger.info("Neo4j connection closed")

    async def verify_connectivity(self) -> bool:
        """
        Verify Neo4j connection is working.

        Returns:
            True if connection verified, False otherwise
        """
        try:
            if self.use_async:
                await self._driver.verify_connectivity()
            else:
                self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified")
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {str(e)}")
            return False

    async def setup_schema(self) -> bool:
        """
        Set up Neo4j schema with constraints and indexes.

        Creates unique constraints for entity_id on Level1 and Level2 nodes,
        and indexes on name properties for better query performance.

        Returns:
            True if schema setup successful, False otherwise
        """
        # Define schema creation queries
        schema_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level1) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level2) REQUIRE n.entity_id IS UNIQUE",
            "CREATE INDEX level1_name_idx IF NOT EXISTS FOR (n:Level1) ON (n.name)",
            "CREATE INDEX level2_name_idx IF NOT EXISTS FOR (n:Level2) ON (n.name)",
        ]

        try:
            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    for query in schema_queries:
                        try:
                            await session.run(query)
                            logger.info(
                                f"Successfully executed schema query: {query}")
                        except Neo4jError as ne:
                            # Handle vector index not supported gracefully
                            if "vector index" in str(ne).lower():
                                logger.warning(
                                    f"Vector index not supported: {str(ne)}")
                            else:
                                logger.error(
                                    f"Neo4j error executing schema query: {str(ne)}")
                                raise
            else:
                # Synchronous version
                with self._driver.session(database=self.database) as session:
                    for query in schema_queries:
                        try:
                            session.run(query)
                            logger.info(
                                f"Successfully executed schema query: {query}")
                        except Neo4jError as ne:
                            if "vector index" in str(ne).lower():
                                logger.warning(
                                    f"Vector index not supported: {str(ne)}")
                            else:
                                logger.error(
                                    f"Neo4j error executing schema query: {str(ne)}")
                                raise

            logger.info("Neo4j schema setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up Neo4j schema: {str(e)}")
            return False

    async def clear_database(self) -> bool:
        """
        Delete all data from Neo4j database.

        WARNING: This operation removes all nodes and relationships!

        Returns:
            True if clearing successful, False otherwise
        """
        try:
            # Cypher query to delete all nodes and relationships
            query = "MATCH (n) DETACH DELETE n"

            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    await session.run(query)
            else:
                with self._driver.session(database=self.database) as session:
                    session.run(query)

            logger.info("Successfully cleared all data from Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Error clearing Neo4j database: {str(e)}")
            return False

    async def import_data(self, data: Dict[str, Any], **kwargs) -> bool:
        """
        Import knowledge graph data into Neo4j.

        Args:
            data: Knowledge graph data containing nodes and edges
            **kwargs: Additional import parameters

        Returns:
            True if import successful, False otherwise
        """
        return await self.import_knowledge_graph(data)

    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute query in Neo4j.

        Args:
            query_params: Dictionary containing query and params

        Returns:
            List of query results
        """
        cypher_query = query_params.get('query', '')
        params = query_params.get('params', {})
        return await self.execute_cypher(cypher_query, params)

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Neo4j graph.

        Returns:
            Dictionary containing node and relationship counts by level
        """
        # Define statistics queries
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) AS count",
            "level1_nodes": "MATCH (n:Level1) RETURN count(n) AS count",
            "level2_nodes": "MATCH (n:Level2) RETURN count(n) AS count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
            "level1_relationships": "MATCH (:Level1)-[r]->(:Level1) RETURN count(r) AS count",
            "level2_relationships": "MATCH (:Level2)-[r]->(:Level2) RETURN count(r) AS count",
            "cross_level_relationships": "MATCH (:Level1)-[r]->(:Level2) RETURN count(r) AS count"
        }

        stats = {}

        try:
            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    # Execute each statistics query
                    for key, query in queries.items():
                        result = await session.run(query)
                        record = await result.single()
                        stats[key] = record["count"] if record else 0
            else:
                # Synchronous version
                with self._driver.session(database=self.database) as session:
                    for key, query in queries.items():
                        result = session.run(query)
                        record = result.single()
                        stats[key] = record["count"] if record else 0

            logger.info(f"Graph statistics retrieved: {json.dumps(stats)}")
            return stats
        except Exception as e:
            logger.error(f"Error getting graph statistics: {str(e)}")
            return {}

    # Graph database specific methods
    async def import_nodes(self, nodes: List[Dict[str, Any]], label: str = None, batch_size: int = 1000, **kwargs) -> bool:
        """
        Import nodes into Neo4j with label and batch processing.

        Args:
            nodes: List of node dictionaries to import
            label: Node label (Level1 or Level2)
            batch_size: Number of nodes to process in each batch
            **kwargs: Additional parameters

        Returns:
            True if import successful, False otherwise
        """
        if not nodes:
            logger.warning(f"No {label} nodes to import")
            return True

        try:
            # Check for duplicate entity_ids
            entity_ids = [node.get('entity_id') for node in nodes]
            duplicate_ids = [id for id in set(
                entity_ids) if entity_ids.count(id) > 1]
            if duplicate_ids:
                logger.warning(
                    f"Found {len(duplicate_ids)} duplicate entity_ids in the input data")

            # Log sample node structure for debugging
            if nodes:
                sample_node = {
                    k: v for k, v in nodes[0].items() if k != 'vector_embedding'}
                logger.info(f"Sample node structure: {sample_node}")

            # Process nodes in batches for better performance
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]

                # Create Cypher query for importing nodes
                query = f"""
                UNWIND $nodes AS node
                MERGE (n:{label} {{entity_id: node.entity_id}})
                SET n = node
                """

                # Clean nodes for Neo4j compatibility
                cleaned_batch = self._clean_node_properties(batch)

                if self.use_async:
                    async with self._driver.session(database=self.database) as session:
                        try:
                            await session.run(query, nodes=cleaned_batch)
                            logger.info(
                                f"Successfully executed node import query for batch of {len(cleaned_batch)} nodes")
                        except Exception as e:
                            logger.error(
                                f"Error during node import transaction: {str(e)}")
                            if cleaned_batch:
                                logger.error(
                                    f"Sample node that caused error: {cleaned_batch[0].get('entity_id', 'unknown')}")
                            raise
                else:
                    # Synchronous version
                    with self._driver.session(database=self.database) as session:
                        session.run(query, nodes=cleaned_batch)
                        logger.info(
                            f"Successfully executed node import query for batch of {len(cleaned_batch)} nodes")

                logger.info(
                    f"Imported batch of {len(batch)} {label} nodes ({i+1}-{i+len(batch)} of {len(nodes)})")

            logger.info(f"Successfully imported {len(nodes)} {label} nodes")
            return True
        except Exception as e:
            logger.error(f"Error importing {label} nodes: {str(e)}")
            return False

    async def import_relationships(self, relationships: List[Dict[str, Any]],
                                   source_label: str = None, target_label: str = None,
                                   batch_size: int = 1000, **kwargs) -> bool:
        """
        Import relationships into Neo4j with optimized performance.

        Args:
            relationships: List of relationship dictionaries
            source_label: Label for source nodes
            target_label: Label for target nodes
            batch_size: Number of relationships to process in each batch
            **kwargs: Additional parameters

        Returns:
            True if import successful, False otherwise
        """
        if not relationships:
            logger.warning(
                f"No relationships to import from {source_label} to {target_label}")
            return True

        try:
            # Track import statistics
            total_processed = 0
            total_imported = 0
            total_skipped = 0

            # Log sample relationship structure for debugging
            if relationships:
                logger.info(
                    f"Sample relationship structure: {relationships[0]}")

            # Pre-process relationships to filter duplicates and invalid entries
            filtered_relationships = []
            seen_pairs = set()

            for rel in relationships:
                # Extract required relationship properties
                source_id = rel.get('source_id')
                target_id = rel.get('target_id')

                # Skip relationships with missing required fields
                if not source_id or not target_id:
                    total_skipped += 1
                    continue

                # Check for duplicates within this batch
                rel_key = (source_id, target_id, rel.get('type', 'RELATES_TO'))
                if rel_key in seen_pairs:
                    total_skipped += 1
                    continue

                seen_pairs.add(rel_key)
                filtered_relationships.append(rel)

            logger.info(
                f"Pre-processing: {len(filtered_relationships)} unique relationships after filtering duplicates")

            # Process relationships in batches
            for i in range(0, len(filtered_relationships), batch_size):
                batch = filtered_relationships[i:i + batch_size]

                # Clean relationships for Neo4j compatibility
                cleaned_batch = self._clean_relationship_properties(batch)

                # Group relationships by type for better performance
                relationships_by_type = {}
                for rel in cleaned_batch:
                    rel_type = rel.get('type', 'RELATES_TO')
                    if rel_type not in relationships_by_type:
                        relationships_by_type[rel_type] = []
                    relationships_by_type[rel_type].append(rel)

                # Process each relationship type in separate query
                for rel_type, type_batch in relationships_by_type.items():
                    import_query = f"""
                    UNWIND $relationships AS rel
                    MATCH (source:{source_label} {{entity_id: rel.source_id}})
                    MATCH (target:{target_label} {{entity_id: rel.target_id}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    ON CREATE SET r = rel, r._imported_at = timestamp()
                    ON MATCH SET r = rel, r._updated_at = timestamp()
                    RETURN count(r) as count
                    """

                    try:
                        if self.use_async:
                            async with self._driver.session(database=self.database) as session:
                                result = await session.run(import_query, relationships=type_batch)
                                summary = await result.consume()

                                # Get count of relationships created
                                if hasattr(summary, 'counters'):
                                    created = summary.counters.relationships_created
                                    total_imported += created
                                    logger.info(
                                        f"Successfully imported {created} new {rel_type} relationships")
                                else:
                                    # Fallback if summary counters not available
                                    logger.info(
                                        f"Imported batch of {rel_type} relationships (count unknown)")
                                    total_imported += len(type_batch)
                    except Exception as e:
                        logger.error(
                            f"Error importing {rel_type} relationships: {str(e)}")
                        # Continue with next batch despite errors

                total_processed += len(batch)
                logger.info(
                    f"Progress: {total_processed}/{len(filtered_relationships)} relationships processed")

                # Add small delay between batches to reduce database load
                if i + batch_size < len(filtered_relationships):
                    await asyncio.sleep(0.1)

            logger.info(
                f"Import summary: {total_processed} processed, ~{total_imported} imported, {total_skipped} skipped")
            return True
        except Exception as e:
            logger.error(
                f"Error importing relationships from {source_label} to {target_label}: {str(e)}")
            return False

    async def execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute Cypher query against Neo4j database.

        Args:
            query: Cypher query string
            params: Parameters for the query

        Returns:
            List of query result records as dictionaries
        """
        params = params or {}

        try:
            if self.use_async:
                async with self._driver.session(database=self.database) as session:
                    result = await session.run(query, params)
                    records = await result.data()
            else:
                # Synchronous version
                with self._driver.session(database=self.database) as session:
                    result = session.run(query, params)
                    records = result.data()

            logger.info(
                f"Successfully executed query: {query[:100]}... with {len(records)} results")
            return records
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    # Helper methods from original code
    async def import_knowledge_graph(self, graph_data: Dict[str, Any]) -> bool:
        """
        Import complete knowledge graph into Neo4j.

        Handles nodes and relationships import in proper order:
        1. Setup schema
        2. Import Level 1 nodes
        3. Import Level 2 nodes
        4. Import relationships by type and level

        Args:
            graph_data: Dictionary containing nodes and edges

        Returns:
            True if import successful, False otherwise
        """
        try:
            # Setup schema first
            schema_success = await self.setup_schema()
            if not schema_success:
                logger.warning(
                    "Schema setup had issues, but continuing with import")

            # Split nodes by knowledge level
            level1_nodes = [node for node in graph_data.get(
                'nodes', []) if node.get('knowledge_level') == 1]
            level2_nodes = [node for node in graph_data.get(
                'nodes', []) if node.get('knowledge_level') == 2]

            # Import Level 1 nodes first
            level1_success = await self.import_nodes(level1_nodes, 'Level1')

            # Import Level 2 nodes
            level2_success = await self.import_nodes(level2_nodes, 'Level2')

            if not level1_success or not level2_success:
                logger.error(
                    "Failed to import all nodes, aborting relationship import")
                return False

            # Split relationships by level and type
            level1_edges = [edge for edge in graph_data.get('edges', [])
                            if edge.get('knowledge_level') == 1 and edge.get('type') != 'REFERENCES']

            level2_edges = [edge for edge in graph_data.get('edges', [])
                            if edge.get('knowledge_level') == 2]

            cross_level_edges = [edge for edge in graph_data.get('edges', [])
                                 if edge.get('type') == 'REFERENCES']

            # Import relationships in order
            level1_rel_success = await self.import_relationships(level1_edges, 'Level1', 'Level1')
            level2_rel_success = await self.import_relationships(level2_edges, 'Level2', 'Level2')
            cross_level_success = await self.import_relationships(cross_level_edges, 'Level1', 'Level2')

            if not level1_rel_success or not level2_rel_success or not cross_level_success:
                logger.warning("Some relationships failed to import")

            logger.info("Knowledge graph import completed")

            # Log import statistics
            stats = {
                "nodes": {
                    "level1": len(level1_nodes),
                    "level2": len(level2_nodes),
                    "total": len(level1_nodes) + len(level2_nodes)
                },
                "relationships": {
                    "level1": len(level1_edges),
                    "level2": len(level2_edges),
                    "cross_level": len(cross_level_edges),
                    "total": len(level1_edges) + len(level2_edges) + len(cross_level_edges)
                }
            }

            logger.info(f"Import statistics: {json.dumps(stats)}")

            return True
        except Exception as e:
            logger.error(f"Error importing knowledge graph: {str(e)}")
            return False

    def _clean_relationship_properties(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean relationship properties for Neo4j compatibility.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            List of cleaned relationship dictionaries
        """
        cleaned_relationships = []

        for rel in relationships:
            cleaned_rel = rel.copy()

            # Remove None values
            cleaned_rel = {k: v for k, v in cleaned_rel.items()
                           if v is not None}

            # Generate relationship ID if missing
            if 'relationship_id' not in cleaned_rel:
                cleaned_rel['relationship_id'] = f"rel_{uuid.uuid4().hex[:8]}"

            # Ensure required fields exist
            required_fields = ['source_id', 'target_id', 'type']
            missing_fields = [
                field for field in required_fields if field not in cleaned_rel]
            if missing_fields:
                logger.warning(
                    f"Relationship missing required fields: {missing_fields}. Relationship: {cleaned_rel}")

                # Try to fill missing fields from alternative properties
                if 'source_id' not in cleaned_rel and 'source' in cleaned_rel:
                    cleaned_rel['source_id'] = cleaned_rel['source']

                if 'target_id' not in cleaned_rel and 'target' in cleaned_rel:
                    cleaned_rel['target_id'] = cleaned_rel['target']

                if 'type' not in cleaned_rel:
                    cleaned_rel['type'] = 'RELATES_TO'

            # Skip relationship if still missing required fields
            if not cleaned_rel.get('source_id') or not cleaned_rel.get('target_id'):
                logger.warning(
                    f"Skipping relationship with missing source_id or target_id after cleanup: {cleaned_rel}")
                continue

            # Handle empty vector embedding
            if 'vector_embedding' in cleaned_rel and not cleaned_rel['vector_embedding']:
                cleaned_rel.pop('vector_embedding')

            # Ensure keywords is a list
            if 'keywords' in cleaned_rel and not isinstance(cleaned_rel['keywords'], list):
                cleaned_rel['keywords'] = list(
                    cleaned_rel['keywords']) if cleaned_rel['keywords'] else []

            # Ensure all keys are strings (Neo4j requirement)
            cleaned_rel = {str(k): v for k, v in cleaned_rel.items()}

            cleaned_relationships.append(cleaned_rel)

        return cleaned_relationships

    def _clean_node_properties(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean node properties for Neo4j compatibility.

        Args:
            nodes: List of node dictionaries

        Returns:
            List of cleaned node dictionaries
        """
        cleaned_nodes = []

        for node in nodes:
            cleaned_node = node.copy()

            # Remove None values
            cleaned_node = {k: v for k,
                            v in cleaned_node.items() if v is not None}

            # Verify required entity_id field
            if 'entity_id' not in cleaned_node:
                logger.warning(
                    f"Node missing entity_id, skipping: {cleaned_node.get('entity_name', 'unknown')}")
                continue

            # Set default entity_type if missing
            if 'entity_type' not in cleaned_node:
                cleaned_node['entity_type'] = 'CONCEPT'

            # Remove empty vector_embedding
            if 'vector_embedding' in cleaned_node and not cleaned_node['vector_embedding']:
                cleaned_node.pop('vector_embedding')

            # Ensure all keys are strings (Neo4j requirement)
            cleaned_node = {str(k): v for k, v in cleaned_node.items()}

            cleaned_nodes.append(cleaned_node)

        return cleaned_nodes


# Utility functions
async def import_graph_from_file(file_path: str, neo4j_client: Neo4jClient) -> bool:
    """
    Import knowledge graph from JSON file into Neo4j.

    Args:
        file_path: Path to JSON file containing graph data
        neo4j_client: Initialized Neo4jClient instance

    Returns:
        True if import successful, False otherwise
    """
    try:
        # Load graph data from file
        with open(file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        # Validate graph data structure
        if 'nodes' not in graph_data or 'edges' not in graph_data:
            logger.error(f"Invalid graph data: missing 'nodes' or 'edges' key")
            return False

        # Import graph into Neo4j
        success = await neo4j_client.import_knowledge_graph(graph_data)

        if success:
            # Get statistics after import
            stats = await neo4j_client.get_statistics()
            logger.info(
                f"Import completed. Graph statistics: {json.dumps(stats)}")

        return success
    except Exception as e:
        logger.error(f"Error importing graph from file: {str(e)}")
        return False


# Main execution example
async def main():
    """
    Example main function demonstrating Neo4jClient usage.
    Shows basic operations: connect, setup schema, import data, query.
    """
    # Initialize client with environment variables
    client = Neo4jClient()

    try:
        # Verify connectivity
        connected = await client.verify_connectivity()
        if not connected:
            logger.error("Failed to connect to Neo4j, exiting")
            return

        # Setup schema
        schema_success = await client.setup_schema()
        if not schema_success:
            logger.warning("Schema setup had issues")

        # Example: Load graph data from a file
        graph_path = "output/combined_graph.json"
        import_success = await import_graph_from_file(graph_path, client)

        if import_success:
            logger.info("Graph successfully imported")

            # Example: Execute a sample query
            query_result = await client.execute_cypher(
                "MATCH (n:Level1)-[r:REFERENCES]->(m:Level2) RETURN n.name, m.name LIMIT 5"
            )
            logger.info(f"Sample cross-level references: {query_result}")

    finally:
        # Always close the Neo4j connection
        await client.close()


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Run the main function
    asyncio.run(main())
