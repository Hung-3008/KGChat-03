import sys
import os
import json
from unittest.mock import MagicMock

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.graph_extractor.edge_extractor import EdgeExtractor
from backend.graph_extractor.schema import ValidatedEntity

# Mock LLM Client
class MockLLMClient:
    def generate(self, prompt, format=None):
        print("\n--- Generated Prompt ---")
        print(prompt)
        print("------------------------\n")
        return {"edges": []}

def main():
    print("Initializing EdgeExtractor...")
    llm_client = MockLLMClient()
    extractor = EdgeExtractor(llm_client=llm_client, model_name="test-model")
    
    if extractor.biosyn:
        print("BioSyn initialized successfully.")
    else:
        print("BioSyn failed to initialize.")
        
    text = "The patient suffered from myocardial infarction."
    nodes = [
        {"name": "myocardial infarction", "semantic_type": "Disease_or_Syndrome", "mention": "myocardial infarction"}
    ]
    
    print("Running extraction...")
    extractor.extract(text, nodes, file_name="test_file")
    print("Extraction complete.")

if __name__ == "__main__":
    main()
