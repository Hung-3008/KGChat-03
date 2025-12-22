"""
Dual-Level Knowledge Graph Retriever

This module retrieves relevant information from the two-level knowledge graph
based on high-level and low-level keywords extracted from user queries.
It uses vector similarity search to find relevant concepts in Level 1,
then traverses connections to more specific Level 2 nodes.
"""
import logging
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union, Tuple

from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.utils.logging import get_logger
from backend.pipeline_prompts import PROMPTS
from backend.retrieval.prompt_self_refine import prompt_evaluate, prompt_evaluate_relation, prompt_score_entity

# Configure logger
logger = get_logger(__name__)


async def retrieve_from_knowledge_graph(
    high_level_keywords: List[str],
    low_level_keywords: List[str],
    neo4j_client: Neo4jClient,
    embedding_client: Any,
    qdrant_client: Optional[VectorDBClient] = None,
    top_k: int = 5,
    max_distance: float = 0.8,
    similarity_threshold: float = 0.7
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve relevant information from the two-level knowledge graph.

    Args:
        high_level_keywords: High-level keywords from the query
        low_level_keywords: Low-level keywords from the query
        neo4j_client: Neo4j database client
        embedding_client: Embedding client for generating embeddings (e.g., TransformerEncoder, GeminiEmbeddingWrapper)
        qdrant_client: Optional Vector database client for similarity search
        top_k: Number of top results to retrieve for each keyword
        max_distance: Maximum vector distance for retrievals
        similarity_threshold: Minimum similarity score threshold

    Returns:
        Dictionary with retrieved context from both knowledge graph levels
    """
    # retrieval_context = {
    #     "level1_nodes": [],
    #     "level2_nodes": [],
    #     "relationships": [],
    #     "sources": [],
    #     "combined_text": ""
    # }

    if not high_level_keywords and not low_level_keywords:
        logger.warning("No keywords provided for knowledge graph retrieval")
        return [], [], []

    logger.info(
        f"Retrieving knowledge with high-level keywords: {high_level_keywords}")
    logger.info(
        f"Retrieving knowledge with low-level keywords: {low_level_keywords}")

    all_keywords = high_level_keywords + low_level_keywords

    try:
        # STEP 1: Try exact text matching first for better precision
        exact_match_entities = await retrieve_level1_nodes_by_text(
            all_keywords,
            neo4j_client,
            top_k=top_k
        )

        logger.info(
            f"Found {len(exact_match_entities)} nodes by exact text matching")

        # STEP 2: Generate embeddings for all keywords
        # Use embed_async if available, otherwise fallback to embed (sync or async)
        if hasattr(embedding_client, 'embed_async'):
            # Prefer async method (e.g., TransformerEncoder.embed_async)
            embeddings = await embedding_client.embed_async(all_keywords)
        elif hasattr(embedding_client, 'embed'):
            # Check if embed is async or sync
            import inspect
            embed_method = getattr(embedding_client, 'embed')
            if inspect.iscoroutinefunction(embed_method):
                # embed is async (e.g., GeminiEmbeddingWrapper.embed)
                embeddings = await embedding_client.embed(all_keywords)
            else:
                # embed is sync (e.g., TransformerEncoder.embed), run in executor
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: embedding_client.embed(all_keywords)
                )
        else:
            raise ValueError(
                "Embedding client must have 'embed_async' or 'embed' method")

        # STEP 3: Retrieve relevant Level 1 nodes using vector similarity
        embedding_entities = await retrieve_level1_nodes(
            embeddings,
            qdrant_client,
            neo4j_client,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        logger.info(
            f"Found {len(embedding_entities)} nodes by embedding similarity")

        # STEP 4: Combine results, prioritizing exact matches
        # Create a dict to avoid duplicates, with exact matches having higher priority
        combined_entities = {}

        # Add exact matches first (higher priority)
        for entity in exact_match_entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if entity_id:
                # Mark as exact match with higher score
                entity["similarity_score"] = 1.0
                entity["match_type"] = "exact"
                combined_entities[entity_id] = entity

        # Add embedding matches (only if not already in exact matches)
        for entity in embedding_entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if entity_id and entity_id not in combined_entities:
                entity["match_type"] = "embedding"
                combined_entities[entity_id] = entity

        # Convert back to list and sort by similarity score
        level1_entities = list(combined_entities.values())
        level1_entities.sort(key=lambda x: x.get(
            "similarity_score", 0), reverse=True)

        # Limit to top results
        max_total_nodes = top_k * 3
        if len(level1_entities) > max_total_nodes:
            level1_entities = level1_entities[:max_total_nodes]

        logger.info(
            f"Combined total: {len(level1_entities)} unique Level 1 nodes")

        # # Store Level 1 nodes in context
        # retrieval_context["level1_nodes"] = level1_entities

        # STEP 3: For each retrieved Level 1 node, find referenced Level 2 nodes
        level2_entities, relationships = await retrieve_level2_references(
            level1_entities,
            neo4j_client,
            max_references=top_k
        )

        # # Store Level 2 nodes and relationships in context
        # retrieval_context["level2_nodes"] = level2_entities
        # retrieval_context["relationships"] = relationships

        # # STEP 4: Format the retrieved information into a combined text
        # combined_text = format_retrieval_results(
        #     level1_entities,
        #     level2_entities,
        #     relationships
        # )

        # retrieval_context["combined_text"] = combined_text

        return level1_entities, level2_entities, relationships

    except Exception as e:
        logger.error(f"Error during knowledge graph retrieval: {str(e)}")
        return [], [], []


async def get_relationships_lv1_to_lv1(list_ids: List[str], neo4j_client: Neo4jClient) -> List[Dict[str, Any]]:
    try:
        query = """
        MATCH (s:Level1)-[r]->(o:Level1)
        WHERE s.id IN $ids AND o.id IN $ids
        RETURN s.id AS subject_id, s.name AS subject_name, 
               type(r) AS predicate, 
               o.id AS object_id, o.name AS object_name
        """
        results = await neo4j_client.execute_query(
            query,
            {"ids": list_ids}
        )
        relationships = []
        for result in results:
            source_name = result.get("subject_name", "Unknown")
            target_name = result.get("object_name", "Unknown")
            relationship_type = result.get("predicate", "RELATED_TO")
            
            rel_data = {
                "source_id": result.get("subject_id"),
                "target_id": result.get("object_id"),
                "target_name": target_name,
                "source_name": source_name,
                "type": relationship_type,
                "description": f"{source_name} {relationship_type.lower()} {target_name}"
            }
            relationships.append(rel_data)
        return relationships
    except Exception as e:
        logger.error(f"Error retrieving Level 1 to Level 1 relationships: {str(e)}")
        return []

async def get_connected_nodes(node_id, relation, neo4j_client: Neo4jClient):
    query = """
    MATCH (n:Level1 {id: $node_id})-[r]-(m:Level1)
    WHERE type(r) = $relation
    RETURN m.name as name
    """
    results = await neo4j_client.execute_query(query, {"node_id": node_id, "relation": relation})
    return [record["name"] for record in results]

async def get_node_metadata(node_name: str, neo4j_client: Neo4jClient) -> Dict[str, Any]:
    query = """
    MATCH (n:Level1)
    WHERE toLower(n.name) = toLower($name)
    RETURN n.id as id, n.name as name, n.entity_type as entity_type, n.description as description
    LIMIT 1
    """
    results = await neo4j_client.execute_query(query, {"name": node_name})
    if results:
        return {
            "id": results[0]["id"],
            "name": results[0]["name"],
            "entity_type": results[0].get("entity_type", "Unknown"),
            "description": results[0].get("description", "")
        }
    return {}

def extract_top_score(response_text: str) -> str:
    """
    Extract the relation/entity with the highest score from the LLM response.
    Expected format: {relation_name (Score: 0.x)}
    """
    import re
    pattern = r"\{([^}]+)\s\(Score:\s*([\d\.]+)\)\}"
    matches = re.findall(pattern, response_text)
    if not matches:
        return ""
    sorted_matches = sorted(matches, key=lambda x: float(x[1]), reverse=True)
    return sorted_matches[0][0].strip()

async def get_relation_of_node(node_id: str, neo4j_client: Neo4jClient) -> List[str]:
    query = """
    MATCH (n:Level1 {id: $node_id})-[r]-()
    RETURN DISTINCT type(r) AS relation
    """
    result = await neo4j_client.execute_query(query, {"node_id": node_id}) 
    relations = [record["relation"] for record in result]
    return relations

async def self_refine(
    level1_nodes: List[Dict[str, Any]],
    level2_nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    gemini_client: Any,
    query: str,
    max_iterations: int = 2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Refine the retrieval results using self-refinement with LLM.
    """
    level1_node_ids = [node["id"] for node in level1_nodes]
    
    # get relation lv1 to lv1
    relations_lv1_to_lv1 = await get_relationships_lv1_to_lv1(level1_node_ids, neo4j_client)
    evaluation_triplets = [r["description"] for r in relations_lv1_to_lv1] + [r["description"] for r in relationships]
    evaluation_triplets = "\n".join(evaluation_triplets)
    
    # Initialize return variables
    relation_cobine = relations_lv1_to_lv1 + relationships
    evaluation_prompt = prompt_evaluate.format(
            query=query,
            triplets=evaluation_triplets
    )
    
    try:
        lv1_nodes_expand = []
        level2_nodes_expand = []
        relations_lv1_to_lv2_expand = []
        relations_expand = []
        # evaluate first step
        evaluation_response = gemini_client.generate(evaluation_prompt)
        is_sufficient = "yes" in evaluation_response.message.lower()
        if is_sufficient:
            logger.info(
                    f"Enough information found after {iteration} iterations")
        else:
            for depth in range(1, max_iterations):
                for item in level1_nodes:
                    # evaluate relation
                    relations = await get_relation_of_node(item["id"], neo4j_client)
                    evaluation_relation_prompt = prompt_evaluate_relation.format(
                        query=query,
                        relations=relations, 
                    )
                    evaluation_relation_response = gemini_client.generate(evaluation_relation_prompt)
                    # top score relation
                    top_relation = extract_top_score(evaluation_relation_response.message)
                    # get entity of relation
                    connected_nodes = await get_connected_nodes(item["id"], top_relation, neo4j_client)
                    # evaluate entity
                    evaluation_entity_prompt = prompt_score_entity.format(
                        query=query,
                        relations=top_relation,
                        old_entities=[n["name"] for n in level1_nodes],
                        entities=connected_nodes
                    )
                    evaluation_entity_response = gemini_client.generate(evaluation_entity_prompt)
                    top_entity = extract_top_score(evaluation_entity_response.message)
                    # get metadata of entity
                    node_metadata = await get_node_metadata(top_entity, neo4j_client)
                    lv1_nodes_expand.append(node_metadata)
                    
                    # relation data lv1_to_lv1
                    rel_data_lv1_to_lv1_expand = [{
                        "source_id": item.get("id"),
                        "target_id": node_metadata.get("id"),
                        "target_name": node_metadata.get("name"),
                        "source_name": item.get("name"),
                        "type": top_relation,
                        "description": f"{item.get('name')} {top_relation.lower()} {node_metadata.get('name')}"
                    }]
                    relations_expand.extend(rel_data_lv1_to_lv1_expand)
                #combine lv1 node
                level1_nodes = level1_nodes + lv1_nodes_expand
                evaluation_triplets_expand = [r["description"] for r in rel_data_lv1_to_lv1_expand] + [r["description"] for r in relations_expand]
                evaluation_triplets_expand = "\n".join(evaluation_triplets_expand)
                relations_lv1_to_lv1 = relations_lv1_to_lv1 + rel_data_lv1_to_lv1_expand # relation lv1 to lv1
                
                # get lv2 nodes
                level2_nodes_expand, relations_lv1_to_lv2_expand = await retrieve_level2_references(
                    lv1_nodes_expand, neo4j_client
                )
                #cobine
                level2_nodes = level2_nodes + level2_nodes_expand
                relationships = relationships + relations_expand # relation lv1 to lv2
                relation_cobine = relations_lv1_to_lv1 + relationships # relation lv1 to lv1 + lv1 to lv2
                evaluation_response = gemini_client.generate(evaluation_prompt)
                is_sufficient = "yes" in evaluation_response.message.lower()
                if is_sufficient:
                    logger.info(
                        f"Enough information found after {iteration} iterations")
                    break 
    except Exception as e:
        logger.error(f"Error in evaluation and expand entities: {str(e)}")
    
    return level1_nodes, level2_nodes, relation_cobine


async def retrieve_level1_nodes_by_text(
    keywords: List[str],
    neo4j_client: Neo4jClient,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve Level 1 nodes using exact text matching on node names.

    Args:
        keywords: List of keywords to search for
        neo4j_client: Neo4j database client
        top_k: Maximum number of results per keyword

    Returns:
        List of retrieved Level 1 node dictionaries
    """
    retrieved_entities = []
    unique_entity_ids = set()

    try:
        for keyword in keywords:
            # Search for nodes where name contains the keyword (case-insensitive)
            # Using CONTAINS for partial matching, or use "= $keyword" for exact match
            query = """
            MATCH (n:Level1)
            WHERE toLower(n.name) CONTAINS toLower($keyword)
            RETURN n.id as id, n.name as name, n.semantic_type as semantic_type,
                   n.cui as cui, n.definition as definition, n.icd as icd, n.level as level
            LIMIT $limit
            """

            try:
                logger.info(
                    f"Searching for Level1 nodes containing: '{keyword}'")
                results = await neo4j_client.execute_query(
                    query,
                    {"keyword": keyword, "limit": top_k}
                )

                logger.info(
                    f"Found {len(results) if results else 0} nodes for keyword '{keyword}'")

                for record in results:
                    entity_id = record.get("id")

                    if not entity_id or entity_id in unique_entity_ids:
                        continue

                    node_data = {
                        "id": entity_id,
                        "entity_id": entity_id,
                        "name": record.get("name", ""),
                        "semantic_type": record.get("semantic_type", ""),
                        "cui": record.get("cui", ""),
                        "definition": record.get("definition", ""),
                        "icd": record.get("icd", ""),
                        "level": record.get("level", "Level 1"),
                        "entity_type": record.get("semantic_type", "CONCEPT"),
                        "description": record.get("definition", ""),
                        "similarity_score": 1.0,  # Perfect match for exact text search
                        "match_type": "text"
                    }

                    retrieved_entities.append(node_data)
                    unique_entity_ids.add(entity_id)
                    logger.debug(
                        f"Found node: {node_data.get('name')} (id: {entity_id})")

            except Exception as e:
                logger.error(
                    f"Error searching for keyword '{keyword}': {str(e)}")

        logger.info(
            f"Retrieved {len(retrieved_entities)} unique Level 1 nodes by text matching")
        return retrieved_entities

    except Exception as e:
        logger.error(f"Error in text-based Level 1 node retrieval: {str(e)}")
        return []


async def retrieve_level1_nodes(
    embeddings: List[List[float]],
    qdrant_client: Any,
    neo4j_client: Neo4jClient,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve Level 1 nodes using vector similarity search.

    Args:
        embeddings: List of embedding vectors for keywords
        qdrant_client: Vector database client for similarity search
        neo4j_client: Neo4j database client for retrieving node data
        top_k: Number of top results to retrieve for each keyword
        similarity_threshold: Minimum similarity score threshold

    Returns:
        List of retrieved Level 1 node dictionaries
    """
    retrieved_entities = []
    unique_entity_ids = set()

    try:
        for embedding in embeddings:
            try:
                # Query Qdrant for similar vectors
                similar_nodes = qdrant_client.query_points(
                    collection_name="kg_lv1_nodes",
                    query=embedding,
                    limit=top_k
                )

                logger.info(
                    f"Found {len(similar_nodes.points) if hasattr(similar_nodes, 'points') else 0} similar Level 1 nodes")

                for node in similar_nodes.points:
                    # Extracting node ID directly from Qdrant point id, which corresponds to the 'id' field in Neo4j
                    # Using node.id directly as the identifier
                    entity_id = str(node.id) if hasattr(
                        node, 'id') and node.id is not None else None
                    similarity_score = getattr(node, 'score', None)

                    if not entity_id:
                        logger.warning(
                            f"Node missing id. "
                            f"Payload keys: {list(node.payload.keys()) if node.payload else 'None'}, "
                            f"Score: {similarity_score}")
                        continue

                    # Apply similarity threshold filter
                    if similarity_score is not None and similarity_score < similarity_threshold:
                        logger.debug(
                            f"Skipping node {entity_id} with similarity score {similarity_score:.4f} "
                            f"below threshold {similarity_threshold}")
                        continue

                    if entity_id in unique_entity_ids:
                        logger.debug(
                            f"Skipping duplicate node_id: {entity_id} (already in unique_entity_ids)")
                        continue

                    # Querying Neo4j to retrieve full node data matching the structure: id, name, semantic_type, cui, definition, icd, level
                    # Using parameterized query first, with fallback to direct string interpolation if needed
                    query = """
                    MATCH (n:Level1 {id: $entity_id})
                    RETURN n.id as id, n.name as name, n.semantic_type as semantic_type,
                           n.cui as cui, n.definition as definition, n.icd as icd, n.level as level
                    """

                    try:
                        logger.info(
                            f"Querying Neo4j with entity_id: {entity_id} (type: {type(entity_id).__name__})")
                        results = await neo4j_client.execute_query(query, {"entity_id": entity_id})
                        logger.info(
                            f"Neo4j query result: {type(results).__name__}, length: {len(results) if results else 0}, "
                            f"result content: {results[:1] if results else 'empty'}")

                        # Fallback to direct string interpolation if parameterized query returns no results
                        if not results or len(results) == 0:
                            # Checking total number of Level1 nodes in database for diagnostic purposes
                            count_query = "MATCH (n:Level1) RETURN count(n) as total"
                            count_result = await neo4j_client.execute_query(count_query)
                            total_nodes = count_result[0].get(
                                'total', 0) if count_result and len(count_result) > 0 else 0
                            logger.warning(
                                f"Node not found with id: {entity_id}. "
                                f"Total Level1 nodes in database: {total_nodes}")

                            # Attempting direct query with string interpolation as fallback
                            logger.warning(
                                f"Attempting direct query with string interpolation for entity_id: {entity_id}")
                            query_direct = f"""
                            MATCH (n:Level1 {{id: '{entity_id}'}})
                            RETURN n.id as id, n.name as name, n.semantic_type as semantic_type,
                                   n.cui as cui, n.definition as definition, n.icd as icd, n.level as level
                            """
                            results_direct = await neo4j_client.execute_query(query_direct)
                            logger.info(
                                f"Direct query result: {type(results_direct).__name__}, length: {len(results_direct) if results_direct else 0}")
                            if results_direct and len(results_direct) > 0:
                                logger.warning(
                                    f"Direct query found node but parameterized query did not. "
                                    f"This may indicate a parameter binding issue.")
                                results = results_direct

                        if results and len(results) > 0:
                            # Creating node data dictionary matching the actual node structure
                            # Neo4j stores: id, name, semantic_type, cui, definition, icd, level
                            # Qdrant payload stores: name, semantic_type
                            node_data = {
                                "id": results[0].get("id", entity_id),
                                # Keeping entity_id for backward compatibility
                                "entity_id": results[0].get("id", entity_id),
                                "name": results[0].get("name", ""),
                                "semantic_type": results[0].get("semantic_type", ""),
                                # CUI code (may be empty)
                                "cui": results[0].get("cui", ""),
                                # Definition (may be empty)
                                "definition": results[0].get("definition", ""),
                                # ICD code (may be empty)
                                "icd": results[0].get("icd", ""),
                                # Level (should be "Level 1")
                                "level": results[0].get("level", "Level 1"),
                                # Keeping entity_type for backward compatibility (using semantic_type)
                                "entity_type": results[0].get("semantic_type", "CONCEPT"),
                                # Using definition as description if available, otherwise empty
                                "description": results[0].get("definition", ""),
                                "similarity_score": similarity_score
                            }

                            retrieved_entities.append(node_data)
                            unique_entity_ids.add(entity_id)
                            logger.debug(
                                f"Retrieved node from Neo4j: {node_data.get('name')} "
                                f"(id: {entity_id}, score: {similarity_score})")
                        else:
                            logger.warning(
                                f"Node not found in Neo4j with id: {entity_id} "
                                f"(score: {similarity_score})")
                    except Exception as query_error:
                        logger.error(
                            f"Error querying Neo4j for entity_id {entity_id}: {str(query_error)}")
                        import traceback
                        logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"Error retrieving similar nodes: {str(e)}")

        # Sort by similarity score
        retrieved_entities.sort(key=lambda x: x.get(
            "similarity_score", 0), reverse=True)

        # Limit to top_k * 2 most relevant nodes overall
        max_nodes = top_k * 2
        if len(retrieved_entities) > max_nodes:
            retrieved_entities = retrieved_entities[:max_nodes]

        logger.info(
            f"Retrieved {len(retrieved_entities)} unique Level 1 nodes")
        return retrieved_entities

    except Exception as e:
        logger.error(f"Error in Level 1 node retrieval: {str(e)}")
        return []


async def retrieve_level2_references(
    level1_nodes: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    max_references: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve Level 2 nodes referenced by Level 1 nodes.

    Args:
        level1_nodes: List of Level 1 node dictionaries
        neo4j_client: Neo4j database client
        max_references: Maximum number of references to retrieve per Level 1 node

    Returns:
        Tuple of (level2_nodes, relationships) lists
    """
    level2_nodes = []
    relationships = []
    unique_level2_ids = set()
    unique_relationship_ids = set()

    try:
        for level1_node in level1_nodes:
            # Using 'id' as the primary identifier matching the actual node structure
            entity_id = level1_node.get("id") or level1_node.get("entity_id")
            if not entity_id:
                continue

            # Retrieving Level 2 nodes connected to this Level 1 node
            # Query matches any relationship type between Level1 and Level2 nodes
            # Returns node properties: id, name, cui, definition, semantic_types, semantic_type, icd, and relationship type
            # Relationship types in database may include IS_A, REFERENCES, or other types
            query = """
            MATCH (l1:Level1 {id: $entity_id})-[r]->(l2:Level2)
            RETURN l2.id AS id, l2.name AS name, l2.cui AS cui, 
                   l2.definition AS definition, l2.semantic_types AS semantic_types,
                   l2.semantic_type AS semantic_type, l2.icd AS icd,
                   type(r) AS relationship_type
            LIMIT $limit
            """

            results = await neo4j_client.execute_query(
                query,
                {"entity_id": entity_id, "limit": max_references}
            )

            for record in results:
                # Creating Level 2 node data dictionary matching the actual node structure
                level2_id = record.get("id", "")
                semantic_types = record.get("semantic_types", [])
                # Single semantic_type field
                semantic_type = record.get("semantic_type", "")
                # Extracting first semantic type if available, defaulting to semantic_type or "CONCEPT"
                entity_type = semantic_types[0] if semantic_types else (
                    semantic_type if semantic_type else "CONCEPT")
                # Extracting actual relationship type from query result
                relationship_type = record.get(
                    "relationship_type", "RELATED_TO")

                level2_data = {
                    "id": level2_id,
                    "entity_id": level2_id,  # Keeping entity_id for backward compatibility
                    "name": record.get("name", "Unknown"),
                    "cui": record.get("cui", ""),
                    "definition": record.get("definition", ""),
                    "semantic_types": semantic_types,  # Array of semantic types
                    "semantic_type": semantic_type,  # Single semantic type field
                    "icd": record.get("icd", ""),  # ICD code
                    "entity_type": entity_type,  # Keeping entity_type for backward compatibility
                    # Using definition as description for backward compatibility
                    "description": record.get("definition", "")
                }

                # Adding Level 2 node to list if not already present
                if level2_id and level2_id not in unique_level2_ids:
                    level2_nodes.append(level2_data)
                    unique_level2_ids.add(level2_id)

                # Creating relationship data dictionary with actual relationship type from database
                rel_id = f"{entity_id}_to_{level2_id}"
                if rel_id and rel_id not in unique_relationship_ids:
                    rel_data = {
                        "source_id": entity_id,
                        "target_id": level2_id,
                        "target_name": record.get("name", "Unknown"),
                        "source_name": level1_node.get("name", "Unknown"),
                        "type": relationship_type,  # Using actual relationship type from database
                        "description": f"{level1_node.get('name', 'Unknown')} {relationship_type.lower()} {record.get('name', 'Unknown')}"
                    }

                    relationships.append(rel_data)
                    unique_relationship_ids.add(rel_id)

        logger.info(
            f"Retrieved {len(level2_nodes)} Level 2 nodes and {len(relationships)} relationships")
        return level2_nodes, relationships

    except Exception as e:
        logger.error(f"Error retrieving Level 2 references: {str(e)}")
        return [], []


async def retrieve_level3_references(
    level2_nodes: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    max_references: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve Level 3 nodes referenced by Level 2 nodes.
    Matches Level2 nodes by their CUI to Level3 nodes where level2_node_id equals the CUI.

    Args:
        level2_nodes: List of Level 2 node dictionaries containing CUI
        neo4j_client: Neo4j database client
        max_references: Maximum number of references to retrieve per Level 2 node

    Returns:
        Tuple of (level3_nodes, relationships) lists
    """
    level3_nodes = []
    relationships = []
    unique_level3_ids = set()
    unique_relationship_ids = set()

    try:
        for level2_node in level2_nodes:
            # Extracting CUI from Level2 node, which should match level2_node_id in Level3
            level2_cui = level2_node.get("cui")
            if not level2_cui:
                logger.debug(
                    f"Level2 node {level2_node.get('name', 'Unknown')} has no CUI, skipping Level3 retrieval")
                continue

            # Retrieving Level 3 nodes where level2_node_id exactly matches the Level2 CUI
            # Query matches Level3 nodes based on exact property match: level2_node_id = CUI
            # Returns all Level3 node properties including patient information
            query = """
            MATCH (l3:Level3)
            WHERE l3.level2_node_id = $level2_cui
            RETURN l3.id AS id, l3.level2_node_id AS level2_node_id,
                   l3.admission_info_json AS admission_info_json,
                   l3.anchor_age AS anchor_age,
                   l3.anchor_year AS anchor_year,
                   l3.anchor_year_group AS anchor_year_group,
                   l3.gender AS gender,
                   l3.medications_json AS medications_json,
                   l3.procedures_json AS procedures_json,
                   l3.relevant_diagnoses_json AS relevant_diagnoses_json,
                   l3.services_json AS services_json,
                   l3.source AS source,
                   l3.subject_id AS subject_id,
                   l3.total_diagnoses_count AS total_diagnoses_count,
                   l3.total_procedures_count AS total_procedures_count,
                   l3.total_services_count AS total_services_count
            LIMIT $limit
            """

            results = []
            relationship_type = "RELATED_TO"

            try:
                results = await neo4j_client.execute_query(
                    query,
                    {"level2_cui": level2_cui, "limit": max_references}
                )
            except Exception as query_error:
                logger.warning(
                    f"Error querying Level3 nodes for CUI {level2_cui}: {query_error}")
                continue

            for record in results:
                # Creating Level 3 node data dictionary with all properties
                level3_id = record.get("id", "")

                level3_data = {
                    "id": level3_id,
                    "entity_id": level3_id,  # Keeping entity_id for backward compatibility
                    "level2_node_id": record.get("level2_node_id", ""),
                    "admission_info_json": record.get("admission_info_json"),
                    "anchor_age": record.get("anchor_age"),
                    "anchor_year": record.get("anchor_year"),
                    "anchor_year_group": record.get("anchor_year_group"),
                    "gender": record.get("gender"),
                    "medications_json": record.get("medications_json"),
                    "procedures_json": record.get("procedures_json"),
                    "relevant_diagnoses_json": record.get("relevant_diagnoses_json"),
                    "services_json": record.get("services_json"),
                    "source": record.get("source"),
                    "subject_id": record.get("subject_id"),
                    "total_diagnoses_count": record.get("total_diagnoses_count"),
                    "total_procedures_count": record.get("total_procedures_count"),
                    "total_services_count": record.get("total_services_count")
                }

                # Adding Level 3 node to list if not already present
                if level3_id and level3_id not in unique_level3_ids:
                    level3_nodes.append(level3_data)
                    unique_level3_ids.add(level3_id)

                # Creating relationship data dictionary
                rel_id = f"{level2_cui}_to_{level3_id}"
                if rel_id and rel_id not in unique_relationship_ids:
                    # Using subject_id as name identifier for Level3
                    level3_name = f"Patient {record.get('subject_id', 'Unknown')}"
                    rel_data = {
                        "source_id": level2_node.get("id", ""),
                        "source_cui": level2_cui,
                        "target_id": level3_id,
                        "target_name": level3_name,
                        "source_name": level2_node.get("name", "Unknown"),
                        "type": relationship_type,
                        "description": f"{level2_node.get('name', 'Unknown')} {relationship_type.lower()} {level3_name}"
                    }

                    relationships.append(rel_data)
                    unique_relationship_ids.add(rel_id)

        logger.info(
            f"Retrieved {len(level3_nodes)} Level 3 nodes and {len(relationships)} relationships")
        return level3_nodes, relationships

    except Exception as e:
        logger.error(f"Error retrieving Level 3 references: {str(e)}")
        return [], []


def format_retrieval_results(
    level1_nodes: List[Dict[str, Any]],
    level2_nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> str:
    """
    Format the retrieval results into a more informative structured text representation.

    Args:
        level1_nodes: List of Level 1 node dictionaries
        level2_nodes: List of Level 2 node dictionaries
        relationships: List of relationship dictionaries

    Returns:
        Formatted text representation of the retrieved information
    """
    # Group Level 2 nodes by the Level 1 nodes that reference them
    level1_to_level2 = {}

    # Group Level 1 nodes by the Level 1 nodes that reference them
    level1_to_level1 = {}

    # Create dictionaries to quickly look up nodes
    level2_by_name = {node.get('name', 'Unknown')
                               : node for node in level2_nodes}
    level1_by_id = {node.get('id') or node.get(
        'entity_id'): node for node in level1_nodes if node.get('id') or node.get('entity_id')}

    # Group relationships by source entity ID
    for rel in relationships:
        source_id = rel.get('source_id')
        target_id = rel.get('target_id')
        target_name = rel.get('target_name')

        if source_id and target_name:
            # Check if target is a Level 1 node
            if target_id and target_id in level1_by_id:
                # This is a Level1-to-Level1 relationship
                if source_id not in level1_to_level1:
                    level1_to_level1[source_id] = []

                level1_to_level1[source_id].append({
                    'name': target_name,
                    'node': level1_by_id[target_id],
                    'relationship': rel
                })
            # Check if target is a Level 2 node
            elif target_name in level2_by_name:
                # This is a Level1-to-Level2 relationship
                if source_id not in level1_to_level2:
                    level1_to_level2[source_id] = []

                level1_to_level2[source_id].append({
                    'name': target_name,
                    'node': level2_by_name[target_name],
                    'relationship': rel
                })

    # Format the text with each Level 1 node and its related Level 2 nodes
    sections = []

    # Add main content section with detailed information
    main_content = []

    for level1_node in level1_nodes:
        # Using 'id' as the primary identifier matching the actual node structure
        entity_id = level1_node.get('id') or level1_node.get('entity_id')
        entity_name = level1_node.get('name', 'Unknown').upper()
        # Use semantic_type if available (actual field in Neo4j), fallback to entity_type for backward compatibility
        entity_type = level1_node.get(
            'semantic_type') or level1_node.get('entity_type', 'Unknown')
        # Description field not present in actual node structure
        entity_desc = level1_node.get('description', '')

        # Add Level 1 node info
        node_section = [
            f"## {entity_name} ({entity_type})",
        ]
        if entity_desc:
            node_section.append(f"{entity_desc}")
        node_section.append("")

        # Add related Level 1 nodes if any
        related_level1_nodes = level1_to_level1.get(entity_id, [])
        if related_level1_nodes:
            node_section.append(f"### Related Level 1 Concepts:")
            for item in related_level1_nodes:
                level1_related_node = item['node']
                rel = item['relationship']
                level1_related_name = level1_related_node.get(
                    'name', 'Unknown')
                level1_related_cui = level1_related_node.get('cui', '')
                level1_related_definition = level1_related_node.get(
                    'definition', '') or level1_related_node.get('description', '')
                level1_related_type = level1_related_node.get(
                    'semantic_type') or level1_related_node.get('entity_type', 'Unknown')
                relationship_type = rel.get('type', 'RELATED_TO')

                # Formatting Level 1 related node information
                cui_str = f" (CUI: {level1_related_cui})" if level1_related_cui else ""
                rel_type_str = f" [{relationship_type}]" if relationship_type != 'RELATED_TO' else ""

                if level1_related_definition:
                    # Truncate very long definitions
                    if len(level1_related_definition) > 300:
                        level1_related_definition = level1_related_definition[:297] + "..."
                    node_section.append(
                        f"* **{level1_related_name}**{cui_str}{rel_type_str} ({level1_related_type}): {level1_related_definition}")
                else:
                    node_section.append(
                        f"* **{level1_related_name}**{cui_str}{rel_type_str} ({level1_related_type})")

            node_section.append("")

        # Add related Level 2 nodes if any
        related_level2_nodes = level1_to_level2.get(entity_id, [])
        if related_level2_nodes:
            node_section.append(f"### Related Level 2 Concepts:")
            for item in related_level2_nodes:
                level2_node = item['node']
                level2_name = level2_node.get('name', 'Unknown')
                level2_cui = level2_node.get('cui', '')
                level2_definition = level2_node.get(
                    'definition', '') or level2_node.get('description', '')
                semantic_types = level2_node.get('semantic_types', [])
                level2_type = semantic_types[0] if semantic_types else level2_node.get(
                    'entity_type', 'CONCEPT')

                # Formatting Level 2 node information
                # Displaying CUI if available
                cui_str = f" (CUI: {level2_cui})" if level2_cui else ""

                if level2_definition:
                    # Truncate very long definitions
                    if len(level2_definition) > 300:
                        level2_definition = level2_definition[:297] + "..."
                    node_section.append(
                        f"* **{level2_name}**{cui_str} ({level2_type}): {level2_definition}")
                else:
                    node_section.append(
                        f"* **{level2_name}**{cui_str} ({level2_type})")

            node_section.append("")

        main_content.extend(node_section)

    # Create a key concepts summary section
    concept_summary = ["# KEY CONCEPTS", ""]

    # Group Level 1 nodes by entity type
    entity_types = {}
    for node in level1_nodes:
        # Use semantic_type if available (actual field in Neo4j), fallback to entity_type for backward compatibility
        entity_type = node.get('semantic_type') or node.get(
            'entity_type', 'Unknown')
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(node)

    # Add a summary for each entity type
    for entity_type, nodes in entity_types.items():
        concept_summary.append(f"## {entity_type.upper()}S")
        for node in nodes:
            name = node.get('name', 'Unknown')
            desc = node.get('description', '')
            # Creating short description from first sentence or truncating, or displaying only name if no description
            if desc:
                short_desc = desc.split('.')[0] if '.' in desc else desc[:50]
                concept_summary.append(f"* **{name}**: {short_desc}")
            else:
                concept_summary.append(f"* **{name}**")
        concept_summary.append("")

    # Add a relationships summary
    relationship_summary = ["# RELATIONSHIPS", ""]

    # Group relationships by type
    rel_types = {}
    for rel in relationships:
        rel_type = rel.get('type', 'RELATED_TO')
        if rel_type not in rel_types:
            rel_types[rel_type] = []
        rel_types[rel_type].append(rel)

    # Add a summary for each relationship type
    for rel_type, rels in rel_types.items():
        relationship_summary.append(f"## {rel_type}")
        # List only unique source-target pairs to avoid repetition
        unique_pairs = set()
        for rel in rels:
            source = rel.get('source_name', 'Unknown')
            target = rel.get('target_name', 'Unknown')
            pair = f"{source} → {target}"
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                relationship_summary.append(f"* {pair}")
        relationship_summary.append("")

    # Combine all sections
    sections.append("\n".join(concept_summary))
    sections.append("\n".join(relationship_summary))
    sections.append("# DETAILED INFORMATION\n")
    sections.append("\n".join(main_content))

    return "\n".join(sections)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Convert text to a valid filename.

    Args:
        text: Text to convert
        max_length: Maximum length of the filename

    Returns:
        Sanitized filename
    """
    # Remove special symbols from text
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip('_')


def create_rag_prompt(query: str, formatted_text: str, conversation_history: Optional[str] = None) -> str:
    """
    Create RAG prompt ready to be passed to LLM.

    Args:
        query: User's query
        formatted_text: Text formatted from format_retrieval_results
        conversation_history: Conversation history (optional)

    Returns:
        Prompt string ready to be passed to LLM
    """
    # Use prompt template from pipeline_prompts.py
    prompt_template = PROMPTS.get("test_full_rag_prompt", "")

    # Format conversation history
    history_text = conversation_history if conversation_history else "(No previous conversation)"

    # Format prompt with values
    prompt = prompt_template.format(
        query=query,
        formatted_text=formatted_text,
        conversation_history=history_text
    )

    return prompt


def save_retrieval_result(
    query: str,
    formatted_text: str,
    result: dict,
    conversation_history: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Save formatted retrieval result to output_pipeline_retrie folder.
    Format according to RAG prompt structure for direct use with LLM.

    Args:
        query: Original query
        formatted_text: Text formatted from format_retrieval_results
        result: Dictionary containing all results
        conversation_history: Conversation history (optional)
        output_dir: Output directory (optional, defaults to output_pipeline_retrie at root)

    Returns:
        Path to saved file or None if error
    """
    try:
        # Import QueryIntent if available (optional dependency)
        query_intent_enum = None
        has_query_intent = False
        try:
            from backend.pipeline.query_analyzer import QueryIntent
            query_intent_enum = QueryIntent
            has_query_intent = True
        except ImportError:
            pass

        # Create output_pipeline_retrie folder if it doesn't exist
        if output_dir is None:
            # Get root path from current file (backend/retrieval/triple_level_retrieval.py)
            # Go up 2 levels to reach root: backend/retrieval -> backend -> root
            current_file = Path(__file__)
            root_path = current_file.parent.parent.parent
            output_dir = root_path / "output_pipeline_retrie"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Create filename based on query and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_sanitized = sanitize_filename(query, max_length=50)
        filename = f"prompt_{timestamp}_{query_sanitized}.txt"
        file_path = output_dir / filename

        # Create RAG prompt ready to be passed to LLM
        rag_prompt = create_rag_prompt(
            query, formatted_text, conversation_history)

        # Create file content with prompt and metadata
        content_parts = []

        # Main prompt section (ready to copy to LLM)
        content_parts.append("=" * 80)
        content_parts.append(
            "PROMPT READY FOR LLM (Copy the section below to use)")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(rag_prompt)
        content_parts.append("")

        # Metadata section (for reference)
        content_parts.append("=" * 80)
        content_parts.append("METADATA (For reference only)")
        content_parts.append("=" * 80)
        content_parts.append(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Intent (if available)
        intent = result.get('intent')
        if intent:
            if has_query_intent and hasattr(intent, 'name'):
                content_parts.append(f"Intent: {intent.name}")
            else:
                content_parts.append(f"Intent: {intent}")
        else:
            content_parts.append("Intent: N/A")

        # Keywords (if available)
        high_level_keywords = result.get('high_level_keywords', [])
        low_level_keywords = result.get('low_level_keywords', [])

        # Display keywords if available (for HEALTHCARE_RELATED or any keywords)
        if high_level_keywords or low_level_keywords:
            # Only display if intent is HEALTHCARE_RELATED (if QueryIntent exists) or if QueryIntent doesn't exist
            should_show_keywords = True
            if has_query_intent and query_intent_enum and intent:
                should_show_keywords = (
                    intent == query_intent_enum.HEALTHCARE_RELATED)

            if should_show_keywords:
                content_parts.append(
                    f"High-level keywords: {', '.join(high_level_keywords)}")
                content_parts.append(
                    f"Low-level keywords: {', '.join(low_level_keywords)}")

        # Retrieval statistics
        if result.get('retrieval_result'):
            retrieval = result['retrieval_result']
            content_parts.append(f"Retrieval Statistics:")
            content_parts.append(
                f"  - Level 1 nodes: {len(retrieval.get('level1_nodes', []))}")
            content_parts.append(
                f"  - Level 2 nodes: {len(retrieval.get('level2_nodes', []))}")
            content_parts.append(
                f"  - Relationships: {len(retrieval.get('relationships', []))}")

        # Write file
        full_content = "\n".join(content_parts)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        logger.info(f"Saved retrieval result to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Error saving retrieval result: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
