import sys
import os
import json
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.de_node_extractor import NodeExtractor
from backend.graph_extractor.graph_extract import GraphExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_node_extractor")

def main():
    config_path = "backend/configs/configs.yml"
    
    # We can reuse GraphExtractor to initialize NodeExtractor
    graph_extractor = GraphExtractor(config_path=config_path)
    node_extractor = graph_extractor.node_extractor
    
    text = "The patient was diagnosed with diabetes mellitus and prescribed metformin. He has a history of hypertension."
    
    print("\n--- Testing NodeExtractor ---")
    
    # We want to inspect the internal steps of extract()
    # Since we can't easily hook into the method without modifying code, 
    # we will rely on the logging I added to de_node_extractor.py earlier (Step 312)
    # or we can call the internal methods if we want.
    
    # Let's try calling extract and see the logs.
    nodes = node_extractor.extract(text, file_name="debug_test")
    
    print(f"\nExtracted Nodes: {len(nodes)}")
    print(json.dumps(nodes, indent=2))

if __name__ == "__main__":
    main()
