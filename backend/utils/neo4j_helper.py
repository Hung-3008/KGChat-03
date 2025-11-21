import os
from neo4j import GraphDatabase
from typing import List, Dict
import logging

logger = logging.getLogger("neo4j_helper")

class Neo4jHelper:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_database(self):
        """Removes all nodes and relationships from the database."""
        with self.driver.session() as session:
            # Delete all relationships first
            session.run("MATCH ()-[r]->() DELETE r")
            # Then delete all nodes
            session.run("MATCH (n) DELETE n")
        logger.info("Cleared all nodes and relationships from Neo4j")

    def drop_constraints(self):
        """Drops unique constraint on Level1 nodes if exists."""
        with self.driver.session() as session:
            # Get all constraints for Level1 nodes
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record for record in result]
            
            # Drop constraints related to Level1 nodes
            for constraint in constraints:
                constraint_name = constraint.get("name")
                # Check if this constraint is for Level1 label
                if constraint_name and "Level1" in str(constraint):
                    try:
                        session.run(f"DROP CONSTRAINT {constraint_name}")
                        logger.info(f"Dropped constraint: {constraint_name}")
                    except Exception as e:
                        logger.warning(f"Could not drop constraint {constraint_name}: {e}")

    def create_index(self):
        """Creates a unique constraint on node_id for Level1 nodes."""
        query = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level1) REQUIRE n.id IS UNIQUE"
        with self.driver.session() as session:
            session.run(query)
        logger.info("Created unique constraint on Level1(id)")

    def insert_nodes(self, nodes: List[Dict]):
        """
        Batch insert nodes.
        Expected node dict: {'id': str, 'name': str, 'semantic_type': str, ...}
        """
        query = """
        UNWIND $nodes AS node
        MERGE (n:Level1 {id: node.id})
        SET n.name = node.name,
            n.semantic_type = node.semantic_type
        """
        with self.driver.session() as session:
            session.run(query, nodes=nodes)

    def insert_edges(self, edges: List[Dict]):
        """
        Batch insert edges.
        Expected edge dict: {'source_id': str, 'target_id': str, 'relation': str, ...}
        """
        query = """
        UNWIND $edges AS edge
        MATCH (s:Level1 {id: edge.source_id})
        MATCH (t:Level1 {id: edge.target_id})
        MERGE (s)-[r:RELATION {type: edge.relation}]->(t)
        """
        # Note: Dynamic relationship types are tricky in Cypher. 
        # For simplicity, we store relation type as a property 'type' on a generic 'RELATION' relationship,
        # OR we can use APOC if available. 
        # A better approach for pure Cypher without APOC for dynamic types is to use APOC or string formatting (risky).
        # Given the requirement, let's use a generic relationship with a type property for now, 
        # OR better: assume 'relation' is the relationship TYPE.
        
        # If 'relation' is the relationship TYPE (e.g. TREATS), we need APOC or string formatting.
        # Let's try APOC approach if available, or fallback to string formatting safely if we trust the input.
        # Since we can't guarantee APOC, and relation types might vary, let's use a fixed relationship type 'RELATED_TO' 
        # and store the actual semantic relation as a property.
        
        query = """
        UNWIND $edges AS edge
        MATCH (s:Level1 {id: edge.source_id})
        MATCH (t:Level1 {id: edge.target_id})
        MERGE (s)-[r:RELATED_TO]->(t)
        SET r.type = edge.relation
        """
        with self.driver.session() as session:
            session.run(query, edges=edges)
