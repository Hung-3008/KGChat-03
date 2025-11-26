
import sys
import os
import json
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("test_level2_nodes")

from backend.graph_extractor.edge_extractor import EdgeExtractor

# Mock LLM Client
class MockLLMClient:
    def generate(self, prompt, format=None):
        return json.dumps({"edges": []})

def main():
    logger.info("Initializing EdgeExtractor...")
    llm_client = MockLLMClient()
    
    try:
        extractor = EdgeExtractor(llm_client=llm_client, model_name="test-model")
    except Exception as e:
        logger.error(f"Failed to initialize EdgeExtractor: {e}")
        return

    if not extractor.biosyn:
        logger.error("BioSyn failed to initialize.")
        return

    # Test case
    text = "The patient has diabetes mellitus and hypertension."
    nodes = [
        {"name": "diabetes mellitus", "semantic_type": "Disease_or_Syndrome", "mention": "diabetes mellitus"},
        {"name": "hypertension", "semantic_type": "Disease_or_Syndrome", "mention": "hypertension"}
    ]
    
    logger.info("Running extraction...")
    edges_result, level2_nodes = extractor.extract(text, nodes, file_name="test_file")
    
    logger.info(f"Level 2 Nodes Created: {len(level2_nodes)}")
    
    missing_defs = 0
    for node in level2_nodes:
        name = node['name']
        cui = node.get('cui')
        definition = node.get('definition')
        
        if definition:
            logger.info(f"✓ {name} ({cui}): Found definition")
        else:
            logger.error(f"✗ {name} ({cui}): Missing definition")
            missing_defs += 1
            
    if missing_defs == 0:
        logger.info("SUCCESS: All Level 2 nodes have definitions.")
    else:
        logger.error(f"FAILURE: {missing_defs} nodes missing definitions.")

if __name__ == "__main__":
    main()
