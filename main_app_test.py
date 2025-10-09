import json
import logging
import os
import uuid
import yaml
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv

from backend.pipeline.chunking.chunker_factory import ChunkerFactory
from backend.pipeline.graph_extraction.extractor_factory import GraphExtractorFactory
from backend.pipeline.graph_extraction.graph_elements import Node, Edge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompt(base_path, file_path):
    """Loads a prompt from a file."""
    with open(os.path.join(base_path, file_path), "r") as f:
        return f.read()

def main():
    load_dotenv()
    # 1. Load configuration and data
    with open("data/test/PMC2749957_result.json", "r") as f:
        document_data = json.load(f)

    # Load prompt configurations from YAML
    with open("backend/pipeline/graph_extraction/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    prompt_settings = config["graph_extraction"]["prompt_settings"]
    prompt_base_path = prompt_settings["base_path"]
    
    # Load node extraction prompts and settings
    node_system_prompt = load_prompt(prompt_base_path, prompt_settings["node_system_prompt"])
    node_user_prompt_template = load_prompt(prompt_base_path, prompt_settings["node_user_prompt"])
    node_feedback_prompt_template = load_prompt(prompt_base_path, prompt_settings["feedback_prompt"])
    node_extraction_mode = prompt_settings["mode"]
    node_extraction_k = prompt_settings["k"]
    model = config["graph_extraction"]["models"][0]
    model_platform = config["graph_extraction"]["provider"]

    # Load edge extraction prompts
    edge_system_prompt = load_prompt(prompt_base_path, prompt_settings["edge_system_prompt"])
    edge_user_prompt_template = load_prompt(prompt_base_path, prompt_settings["edge_user_prompt"])

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Initialize components
    # Initialize Chunker
    chunker = ChunkerFactory.create_chunker("section", min_chunk_tokens=100)

    # Initialize a single Graph Extractor
    graph_extractor = GraphExtractorFactory.create_graph_extractor(
        model_platform=model_platform,
        model_name=model
    )

    # 3. Run pipeline
    chunks = chunker.chunk(document_data, document_metadata={"document_id": "PMC2749957"})
    logging.info(f"Document chunked into {len(chunks)} sections.")

    all_nodes: List[Node] = []
    all_edges: List[Edge] = []
    node_ids = set()

    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk_content = chunk["content"]

        # Create a chunk node
        chunk_id = str(uuid.uuid4())
        chunk_node = Node(id=chunk_id, label="chunk", properties={"content": chunk_content})
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
        #logging.info(f"Extracted {len(nodes)} nodes from chunk {i+1}.")

        chunk_nodes = []
        for node in nodes:
            if node.id not in node_ids:
                chunk_nodes.append(node)
                node_ids.add(node.id)
        all_nodes.extend(chunk_nodes)

        # Create relationships from extracted nodes to the chunk node
        for node in chunk_nodes:
            edge = Edge(source=node.id, target=chunk_node.id, label="source_from")
            all_edges.append(edge)

        # Extract edges
        if chunk_nodes:
            edges = graph_extractor.extract_edges(
                document_content=chunk_content, 
                nodes=chunk_nodes,
                system_prompt=edge_system_prompt,
                user_prompt_template=edge_user_prompt_template
            )
            #logging.info(f"Extracted {len(edges)} edges from chunk {i+1}.")
            all_edges.extend(edges)

    # 4. Save output
    final_nodes = []
    for node in all_nodes:
        node_dict = node.model_dump()
        if node_dict.get("label") != "chunk":
            node_dict.pop("properties", None)
        final_nodes.append(node_dict)

    graph = {
        "nodes": final_nodes,
        "edges": [edge.model_dump() for edge in all_edges],
    }

    output_path = os.path.join(output_dir, "graph.json")
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)

    logging.info(f"Graph saved to {output_path}")

if __name__ == "__main__":
    main() 