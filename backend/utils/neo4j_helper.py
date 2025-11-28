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
        """Drops constraints that might conflict with insertion."""
        with self.driver.session() as session:
            # Get all constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record for record in result]
            
            # Drop constraints related to Level1, Level2, Level3 nodes (except for id uniqueness which we want to keep)
            for constraint in constraints:
                constraint_name = constraint.get("name")
                labels = constraint.get("labelsOrTypes", [])
                properties = constraint.get("properties", [])
                
                # Drop Level2.cui and other potentially conflicting constraints
                # But keep Level1.id, Level2.id, Level3.id for data integrity
                should_drop = False
                
                if "Level2" in labels and "cui" in properties:
                    should_drop = True
                    logger.info(f"Dropping Level2.cui constraint: {constraint_name}")
                elif "Level3" in labels and "case_id" in properties:
                    should_drop = True
                    logger.info(f"Dropping Level3.case_id constraint: {constraint_name}")
                elif "Level3" in labels and "subject_id" in properties:
                    should_drop = True
                    logger.info(f"Dropping Level3.subject_id constraint: {constraint_name}")
                
                if should_drop:
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
            n.semantic_type = node.semantic_type,
            n.icd = node.icd,
            n.definition = node.definition,
            n.cui = node.cui,
            n.level = node.level
        WITH n, node
        CALL apoc.do.when(node.level = 'Level 2', 'SET n:Level2', '', {n:n}) YIELD value
        RETURN count(value) as nodes_processed
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, nodes=nodes)
                summary = result.consume()
                logger.debug(f"Inserted batch: {summary.counters.nodes_created} created, {summary.counters.properties_set} properties set")
                return summary.counters.nodes_created
        except Exception as e:
            logger.error(f"Failed to insert nodes batch: {e}")
            logger.error(f"First node in batch: {nodes[0] if nodes else 'empty batch'}")
            raise

    def insert_edges(self, edges: List[Dict]):
        """
        Batch insert edges.
        Expected edge dict: {'source_id': str, 'target_id': str, 'relation': str, ...}
        """
        query = """
        UNWIND $edges AS edge
        MATCH (s:Level1 {id: edge.source_id})
        MATCH (t:Level1 {id: edge.target_id})
        CALL apoc.merge.relationship(s, edge.relation, {}, {}, t, {})
        YIELD rel
        RETURN count(rel) as edges_processed
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, edges=edges)
                summary = result.consume()
                logger.debug(f"Inserted batch: {summary.counters.relationships_created} relationships created")
                return summary.counters.relationships_created
        except Exception as e:
            logger.error(f"Failed to insert edges batch: {e}")
            logger.error(f"First edge in batch: {edges[0] if edges else 'empty batch'}")
            raise

    def query(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """
        Executes a generic Cypher query and returns the results as a list of dictionaries.
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]
