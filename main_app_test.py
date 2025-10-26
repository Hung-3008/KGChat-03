from backend.pipeline.graph_extraction.graph_elements import Node, Edge
from backend.pipeline.graph_extraction.extractor_factory import GraphExtractorFactory
from backend.pipeline.chunking.chunker_factory import ChunkerFactory
from backend.llm.providers.gemini.gemini_client import GeminiClient
from backend.llm.providers.gemini.gemini_config import GeminiConfig
from data.test.nodes_extraction.preprocessing import load_dataset
import sys
import os
import json
import logging
import csv
import pandas as pd
from typing import List, Dict, Any
import time
import yaml
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import uuid

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_prompt(base_path, file_path):
    """Loads a prompt from a file."""
    with open(os.path.join(base_path, file_path), "r") as f:
        return f.read()


def generate_node_embeddings(nodes: List[Node], gemini_client: GeminiClient) -> List[Node]:
    """Generate embeddings for nodes using Gemini client."""
    try:
        # Extract node texts for embedding
        node_texts = []
        for node in nodes:
            if hasattr(node, 'id') and node.id:
                node_texts.append(str(node.id))
            else:
                node_texts.append("")

        # Generate embeddings using Gemini client
        embeddings = gemini_client.embed(node_texts, max_tries=3)

        # Add embeddings to nodes
        for i, node in enumerate(nodes):
            if hasattr(node, 'properties') and node.properties:
                node.properties['embedding'] = embeddings[i]
            else:
                node.properties = {'embedding': embeddings[i]}

        logging.info(
            f"Generated embeddings for {len(nodes)} nodes using Gemini")
        return nodes

    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        # Return nodes without embeddings if embedding fails
        return nodes


def main():
    load_dotenv()
    # 1. Load configuration
    with open("backend/pipeline/graph_extraction/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load dataset from config
    dataset_name = config["graph_extraction"]["datasets"][0]
    max_samples = config["graph_extraction"]["max_samples"]

    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    if max_samples:
        dataset = dataset[:max_samples]
        logging.info(
            f"Processing {len(dataset)} samples (limited by max_samples={max_samples})")
    else:
        logging.info(f"Processing all {len(dataset)} samples")

    # Process each document in the dataset
    for doc_idx, (document_text, ground_truth) in enumerate(dataset):
        logging.info(f"Processing document {doc_idx + 1}/{len(dataset)}")

        # Create document_data structure for chunker
        # We need to recreate the original structure with separate sections
        document_data = {
            "content_sections": [
                {"header": "FULL_TEXT", "content": document_text}
            ],
            "title": f"Document {doc_idx + 1}"
        }

        # Debug: Check document length
        logging.info(f"Document text length: {len(document_text)} characters")
        logging.info(f"Document text preview: {document_text[:200]}...")

        # Load prompt settings
        prompt_settings = config["graph_extraction"]["prompt_settings"]
        prompt_base_path = prompt_settings["base_path"]

        # Load node extraction prompts and settings
        node_system_prompt = load_prompt(
            prompt_base_path, prompt_settings["node_system_prompt"])
        node_user_prompt_template = load_prompt(
            prompt_base_path, prompt_settings["node_user_prompt"])
        node_feedback_prompt_template = load_prompt(
            prompt_base_path, prompt_settings["feedback_prompt"])
        node_extraction_mode = prompt_settings["mode"]
        node_extraction_k = prompt_settings["k"]
        model = config["graph_extraction"]["models"][0]
        model_platform = config["graph_extraction"]["provider"]

        # Load edge extraction prompts
        edge_system_prompt = load_prompt(
            prompt_base_path, prompt_settings["edge_system_prompt"])
        edge_user_prompt_template = load_prompt(
            prompt_base_path, prompt_settings["edge_user_prompt"])

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # 2. Initialize components
        # Initialize Chunker - use recursive text chunker instead
        chunker = ChunkerFactory.create_chunker(
            "recursive_text", min_chunk_tokens=100)

        # Initialize a single Graph Extractor
        graph_extractor = GraphExtractorFactory.create_graph_extractor(
            model_platform=model_platform,
            model_name=model
        )

        # Initialize Gemini client for embeddings
        gemini_config = GeminiConfig()
        gemini_client = GeminiClient(gemini_config)

        # 3. Run pipeline - pass text directly for recursive chunker
        chunks = chunker.chunk(document_text, document_metadata={
                               "document_id": f"Doc_{doc_idx + 1}"})
        logging.info(f"Document chunked into {len(chunks)} sections.")

        all_nodes: List[Node] = []
        all_edges: List[Edge] = []
        node_ids = set()

        for i, chunk in enumerate(tqdm(chunks, desc=f"Processing chunks for Doc {doc_idx + 1}")):
            chunk_content = chunk["content"]

            # Create a chunk node
            chunk_id = str(uuid.uuid4())
            chunk_node = Node(id=chunk_id, label="chunk",
                              properties={"content": chunk_content})
            all_nodes.append(chunk_node)

            # Extract nodes
            nodes = graph_extractor.extract_nodes(
                document_content=chunk_content,
                system_prompt=node_system_prompt,
                user_prompt_template=node_user_prompt_template,
                mode=node_extraction_mode,
                k=node_extraction_k,
                feedback_prompt_template=node_feedback_prompt_template
            )
            # logging.info(f"Extracted {len(nodes)} nodes from chunk {i+1}.")

            chunk_nodes = []
            for node in nodes:
                if node.id not in node_ids:
                    chunk_nodes.append(node)
                    node_ids.add(node.id)
            all_nodes.extend(chunk_nodes)

            # Skip creating relationships to chunk nodes - we only want entity-to-entity relationships

            # Extract edges between entities in this chunk
            if chunk_nodes:
                edges = graph_extractor.extract_edges(
                    document_content=chunk_content,
                    nodes=chunk_nodes,
                    system_prompt=edge_system_prompt,
                    user_prompt_template=edge_user_prompt_template
                )
                # logging.info(f"Extracted {len(edges)} edges from chunk {i+1}.")
                all_edges.extend(edges)

        # 4. Generate embeddings for entity nodes
        entity_nodes = []
        for node in all_nodes:
            # Check if it's not a chunk node (UUID-based) and is a real entity
            if (not node.id.startswith('f') and  # Not a UUID
                    len(node.id) > 3):  # Not a short ID
                entity_nodes.append(node)

        # Generate embeddings for entity nodes
        if entity_nodes:
            logging.info(
                f"Generating embeddings for {len(entity_nodes)} entity nodes...")
            entity_nodes = generate_node_embeddings(
                entity_nodes, gemini_client)

        # 5. Save output for this document
        final_nodes = []
        entity_node_ids = set()

        # Filter nodes: only keep entity nodes (not chunk nodes with UUIDs)
        for node in entity_nodes:
            node_dict = node.model_dump()
            # Keep properties including embeddings
            final_nodes.append(node_dict)
            entity_node_ids.add(node.id)

        # Filter edges: only keep relationships between entity nodes
        final_edges = []
        for edge in all_edges:
            edge_dict = edge.model_dump()
            # Only include edges between entity nodes
            if (edge_dict["source"] in entity_node_ids and
                    edge_dict["target"] in entity_node_ids):
                final_edges.append(edge_dict)

        graph = {
            "nodes": final_nodes,
            "edges": final_edges,
        }

        output_path = os.path.join(output_dir, f"graph_doc_{doc_idx + 1}.json")
        with open(output_path, "w") as f:
            json.dump(graph, f, indent=2)

        logging.info(
            f"Graph for document {doc_idx + 1} saved to {output_path}")
        logging.info(
            f"Extracted {len(final_nodes)} nodes and {len(final_edges)} edges")


if __name__ == "__main__":
    main()
