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
    def __init__(self, config_path: str = "backend/configs/configs.yml", time_logger: Optional[TimeLogger] = None):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = Path(project_root) / config_path
            
        self.configs = self._load_config()
        self.time_logger = time_logger
        
        # Initialize LLM Client
        self.llm_config = self.configs.get("LLM", {})
        self.llm_client = LLMFactory.create_client(self.llm_config)
        
        # Initialize Components
        self.chunker = SectionChunker()
        
        # Encoder config
        encoder_config = self.configs.get("Encoder", {})
        embedding_model = encoder_config.get("model_name", "intfloat/multilingual-e5-base")
        device = encoder_config.get("device", "cpu")
        
        self.node_extractor = NodeExtractor(
            llm_client=self.llm_client,
            model_name=self.llm_config.get("model", "gpt-like-model"),
            embedding_model=embedding_model,
            device=device,
            time_logger=self.time_logger
        )
        
        self.edge_extractor = EdgeExtractor(
            llm_client=self.llm_client,
            model_name=self.llm_config.get("model", "gpt-like-model"),
            time_logger=self.time_logger
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
            
        logger.info(f"Processing file: {input_path}")
        
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
        for i, chunk in enumerate(chunks):
            # Node Extraction
            if self.time_logger:
                with Timer(self.time_logger, input_file.name, f"Node Extraction (Chunk {i})"):
                    nodes = self.node_extractor.extract(chunk, file_name=input_file.name)
            else:
                nodes = self.node_extractor.extract(chunk, file_name=input_file.name)
                
            for node in nodes:
                node['chunk_id'] = i
                node['source_file'] = input_file.name
            all_nodes.extend(nodes)
            
            # Edge Extraction
            if self.time_logger:
                with Timer(self.time_logger, input_file.name, f"Edge Extraction (Chunk {i})"):
                    edges_result = self.edge_extractor.extract(text=chunk, nodes=nodes, file_name=input_file.name)
            else:
                edges_result = self.edge_extractor.extract(text=chunk, nodes=nodes, file_name=input_file.name)
                
            if edges_result and edges_result.edges:
                for edge in edges_result.edges:
                    edge_dict = edge.dict()
                    edge_dict['chunk_id'] = i
                    edge_dict['source_file'] = input_file.name
                    all_edges.append(edge_dict)
        
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
        print(f"Extraction complete. Results saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Run Graph Extractor")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()
    
    extractor = GraphExtractor()
    extractor.process(args.input, args.output)
