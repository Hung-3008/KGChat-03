import logging
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union, Tuple

from backend.db.neo4j_client import Neo4jClient
from backend.db.vector_db import VectorDBClient
from backend.utils.logging import get_logger
from backend.retrieval.prompt_self_refine import prompt_evaluate, prompt_evaluate_relation, prompt_score_entity


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