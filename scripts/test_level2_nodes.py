import sys
import os
import json
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.edge_extractor import EdgeExtractor
from backend.graph_extractor.schema import ValidatedEntity

# Mock LLM Client
class MockLLMClient:
    def generate(self, prompt, format=None):
        return json.dumps({"edges": []})

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing EdgeExtractor...")
    llm_client = MockLLMClient()
    extractor = EdgeExtractor(llm_client=llm_client, model_name="test-model")
    
    if extractor.biosyn:
        print("BioSyn initialized successfully.")
    else:
        print("BioSyn failed to initialize.")
        return

    text = "The patient has diabetes mellitus."
    nodes = [
        {"name": "diabetes mellitus", "semantic_type": "Disease_or_Syndrome", "mention": "diabetes mellitus"}
    ]
    
    print("Running extraction...")
    edges_result, level2_nodes = extractor.extract(text, nodes, file_name="test_file")
    
    print(f"Level 2 Nodes: {len(level2_nodes)}")
    for node in level2_nodes:
        print(json.dumps(node, indent=2))
        
    print(f"Edges: {len(edges_result.edges)}")
    for edge in edges_result.edges:
        print(edge)

if __name__ == "__main__":
    main()
