import sys
import os
import traceback
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.graph_extract import GraphExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    try:
        extractor = GraphExtractor(config_path="backend/configs/configs.yml")
        input_file = "data/500_samples_pmc/PMC2687513_result.json" # One of the failing files
        
        print(f"Processing {input_file}...")
        nodes, edges = extractor.extract_from_file(input_file)
        print(f"Success! Nodes: {len(nodes)}, Edges: {len(edges)}")
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
