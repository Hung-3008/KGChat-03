import json
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Optional
import sys
import os
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.llm.llm_factory import LLMFactory
from backend.chunkers.section_chunking import SectionChunker
from backend.graph_extractor.de_node_extractor import NodeExtractor
from backend.graph_extractor.edge_extractor import EdgeExtractor
from backend.utils.time_logger import TimeLogger, Timer, setup_logger

logger = setup_logger("graph_extractor")

class GraphExtractor:
    def __init__(self, config_path: str = "backend/configs/configs.yml", time_logger: Optional[TimeLogger] = None, encoder=None, llm_base_url: Optional[str] = None, llm_api_key: Optional[str] = None):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = Path(project_root) / config_path
            
        self.configs = self._load_config()
        self.time_logger = time_logger
        
        # Initialize LLM Client
        self.llm_config = self.configs.get("LLM", {})
        if llm_base_url:
            self.llm_config['base_url'] = llm_base_url
            logger.info(f"Overriding LLM Base URL: {llm_base_url}")
            
        if llm_api_key:
            self.llm_config['api_key'] = llm_api_key
            
        self.llm_client = LLMFactory.create_client(self.llm_config)
        
        # Initialize Components
        self.chunker = SectionChunker()
        
        # Encoder config
        encoder_config = self.configs.get("Encoder", {})
        embedding_model = encoder_config.get("model_name", "intfloat/multilingual-e5-base")
        device = encoder_config.get("device", "cpu")
        
        # Use provided encoder or create new one
        if encoder:
            self.encoder = encoder
        else:
            from backend.encoders.transformer_encoder import TransformerEncoder
            self.encoder = TransformerEncoder(model_name=embedding_model, device=device)
        
        self.node_extractor = NodeExtractor(
            llm_client=self.llm_client,
            model_name=self.llm_config.get("model", "gpt-like-model"),
            embedding_model=embedding_model,
            encoder=self.encoder, # Pass encoder explicitly
            device=device,
            time_logger=self.time_logger
        )
        
        # Qdrant config
        qdrant_config = self.configs.get("Qdrant", {})
        search_batch_size = qdrant_config.get("search_batch_size", 64)
        
        self.edge_extractor = EdgeExtractor(
            llm_client=self.llm_client,
            model_name=self.llm_config.get("model", "gpt-like-model"),
            time_logger=self.time_logger,
            search_batch_size=search_batch_size
        )

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"Warning: Config file not found at {self.config_path}")
            return {}
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def extract_from_file(self, input_path: str) -> tuple[List[Dict], List[Dict]]:
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # logger.info(f"Processing file: {input_path}")
        
        # Step 1: Chunking
        if self.time_logger:
            with Timer(self.time_logger, input_file.name, "Chunking"):
                chunks = self.chunker.chunk(input_file)
        else:
            chunks = self.chunker.chunk(input_file)
            
        all_nodes = []
        all_edges = []
        
        # Step 2 & 3: Node and Edge Extraction per chunk
        # Step 2 & 3: Node and Edge Extraction per chunk
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Determine max workers, default to 3 as it's a reasonable balance
        max_workers = self.configs.get("Create", {}).get("max_parallel_chunks", 3)
        
        def process_chunk(i, chunk):
            chunk_nodes = []
            chunk_edges = []
            
            try:
                # Node Extraction (no timer - logged at stage level)
                nodes = self.node_extractor.extract(chunk, file_name=input_file.name)
                    
                for node in nodes:
                    node['chunk_id'] = i
                    node['source_file'] = input_file.name
                    node['level'] = "Level 1" # Mark as Level 1
                chunk_nodes.extend(nodes)
                
                # Edge Extraction (no timer - logged at stage level)
                logger.info(f"Extracting edges for Chunk {i} with {len(nodes)} nodes")
                edges_result, level2_nodes = self.edge_extractor.extract(text=chunk, nodes=nodes, file_name=input_file.name)
                
                # Process Level 2 Nodes
                for l2_node in level2_nodes:
                    l2_node['chunk_id'] = i
                    l2_node['source_file'] = input_file.name
                    # level is already set in EdgeExtractor
                chunk_nodes.extend(level2_nodes)
                    
                if edges_result and edges_result.edges:
                    for edge in edges_result.edges:
                        edge_dict = edge.dict()
                        edge_dict['chunk_id'] = i
                        edge_dict['source_file'] = input_file.name
                        chunk_edges.append(edge_dict)
                        
                return chunk_nodes, chunk_edges
            except Exception as e:
                import traceback
                logger.error(f"Error extracting chunk {i}: {e}\n{traceback.format_exc()}")
                return [], []

        # Execute chunks in parallel with stage-level timing
        # Note: If memory usage is high, reduce chunk_parallelism in config
        if self.time_logger:
            with Timer(self.time_logger, input_file.name, "All_Chunks_Processing"):
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
                    
                    for future in as_completed(future_to_idx):
                        i = future_to_idx[future]
                        try:
                            c_nodes, c_edges = future.result()
                            all_nodes.extend(c_nodes)
                            all_edges.extend(c_edges)
                        except Exception as e:
                            logger.error(f"Failed to process chunk {i}: {e}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
                
                for future in as_completed(future_to_idx):
                    i = future_to_idx[future]
                    try:
                        c_nodes, c_edges = future.result()
                        all_nodes.extend(c_nodes)
                        all_edges.extend(c_edges)
                    except Exception as e:
                        logger.error(f"Failed to process chunk {i}: {e}")
        
        return all_nodes, all_edges

    def save_nodes(self, nodes: List[Dict], filepath: Path, append: bool = False):
        if not nodes:
            return
            
        # Determine fields
        fieldnames = set()
        for node in nodes:
            fieldnames.update(node.keys())
        
        ordered_fields = ['name', 'semantic_type', 'source_file', 'chunk_id']
        other_fields = [f for f in fieldnames if f not in ordered_fields and f != 'embedding']
        if 'embedding' in fieldnames:
            other_fields.append('embedding')
            
        fieldnames = ordered_fields + other_fields
        
        mode = "a" if append else "w"
        write_header = not append or not filepath.exists() or filepath.stat().st_size == 0
        
        with filepath.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(nodes)
            
    def save_edges(self, edges: List[Dict], filepath: Path, append: bool = False):
        if not edges:
            return
            
        fieldnames = set()
        for edge in edges:
            fieldnames.update(edge.keys())
            
        ordered_fields = ['source', 'relation', 'target', 'source_file', 'chunk_id']
        other_fields = [f for f in fieldnames if f not in ordered_fields]
        fieldnames = ordered_fields + other_fields
        
        mode = "a" if append else "w"
        write_header = not append or not filepath.exists() or filepath.stat().st_size == 0
        
        with filepath.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(edges)

    def process(self, input_path: str, output_dir: str = "output"):
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        nodes, edges = self.extract_from_file(input_path)
        
        self.save_nodes(nodes, output_path / "nodes.csv")
        self.save_edges(edges, output_path / "edges.csv")

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Run Graph Extractor")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()
    
    extractor = GraphExtractor()
    extractor.process(args.input, args.output)