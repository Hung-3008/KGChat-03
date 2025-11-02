import os
import json
import logging
import pandas as pd
from pathlib import Path
import yaml
import warnings

from backend.pipeline.chunking.chunker_factory import ChunkerFactory
from backend.pipeline.graph_extraction.extractor_factory import GraphExtractorFactory
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_files(data_dir, limit=None, batch_size=None):
    json_files = []
    for file_path in Path(data_dir).glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            json_files.append(data)
        if limit is not None and len(json_files) >= limit:
            break
        if batch_size is not None and len(json_files) >= batch_size:
            break
    return json_files

def load_prompt_templates(prompt_base_path):
    prompts = {}
    prompt_files = {
        'node_system_prompt': 'node_system_prompt.txt',
        'node_user_prompt': 'zeroshot_prompt.txt',
        'edge_system_prompt': 'edge_system_prompt.txt',
        'edge_user_prompt': 'edge_cot_prompt.txt'
    }

    for key, filename in prompt_files.items():
        with open(os.path.join(prompt_base_path, filename), 'r', encoding='utf-8') as f:
            prompts[key] = f.read().strip()

    return prompts

def main():
    data_dir = "data/500_samples_pmc"
    config_path = "backend/configs/graph_extraction.yml"
    output_dir = "output"
    batch_size = 5

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading JSON data files...")
    json_data = load_json_files(data_dir, limit=1, batch_size=batch_size)
    logger.info(f"Loaded {len(json_data)} JSON files")

    logger.info("Initializing chunker...")
    chunker = ChunkerFactory.create_chunker("section")

    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Initializing graph extractor...")
    extractor = GraphExtractorFactory.create_graph_extractor(
        model_platform=config['graph_extraction']['provider'],
        model_name=config['graph_extraction']['models'][0]
    )

    logger.info("Loading prompt templates...")
    prompt_base_path = config['graph_extraction']['prompt_settings']['base_path']
    prompts = load_prompt_templates(prompt_base_path)

    all_nodes = []
    all_edges = []

    logger.info("Processing documents...")
    for doc_idx, doc_data in enumerate(tqdm(json_data, desc="Processing documents")):
        document_metadata = {
            "document_id": doc_data.get('file', f"doc_{doc_idx}"),
            "knowledge_level": 1
        }

        chunks = chunker.chunk(doc_data, document_metadata)

        for chunk in chunks:
            content = chunk['content']

            nodes = extractor.extract_nodes(
                document_content=content,
                system_prompt=prompts['node_system_prompt'],
                user_prompt_template=prompts['node_user_prompt'],
                mode=config['graph_extraction']['prompt_settings']['mode'],
                k=config['graph_extraction']['prompt_settings']['k'],
                embedding_provider=config['graph_extraction']['provider'],
                embedding_model=config['graph_extraction']['embedding_model']
            )

            if nodes:
                edges = extractor.extract_edges(
                    document_content=content,
                    nodes=nodes,
                    system_prompt=prompts['edge_system_prompt'],
                    user_prompt_template=prompts['edge_user_prompt']
                )

                for node in nodes:
                    all_nodes.append({
                        'id': node.id,
                        'text': node.properties.get('text', ''),
                        'description': node.properties.get('description', ''),
                        'embedding': json.dumps(node.properties.get('embedding', [])),
                        'document_id': document_metadata['document_id'],
                        'chunk_id': chunk['metadata']['chunk_id']
                    })

                for edge in edges:
                    all_edges.append({
                        'source': edge.source,
                        'target': edge.target,
                        'label': edge.label,
                        'document_id': document_metadata['document_id'],
                        'chunk_id': chunk['metadata']['chunk_id']
                    })

    logger.info(f"Extracted {len(all_nodes)} nodes and {len(all_edges)} edges")

    logger.info("Saving nodes to CSV...")
    nodes_df = pd.DataFrame(all_nodes)
    nodes_df.to_csv(os.path.join(output_dir, 'nodes.csv'), index=False)

    logger.info("Saving edges to CSV...")
    edges_df = pd.DataFrame(all_edges)
    edges_df.to_csv(os.path.join(output_dir, 'edges.csv'), index=False)

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":

    main()

