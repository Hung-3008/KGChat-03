import sys
import os
import json

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from backend.pipeline.chunking.chunker_factory import ChunkerFactory
    from backend.pipeline.graph_extraction.extractor_factory import GraphExtractorFactory
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Please ensure that the backend directory is in your Python path and all dependencies are installed.")
    sys.exit(1)

sample_path = os.path.join("data/test/PMC2749957_result.json")
sample_document = json.load(open(sample_path, 'r')) 

def main():
    print("--- Starting Test ---")

    # --- 2. Chunking ---
    print("\n--- Step 1: Chunking ---")
    chunks = []
    try:
        chunker = ChunkerFactory.create_chunker(
            "section", 
            min_chunk_tokens=20
        )
        chunks = chunker.chunk(sample_document)
        print(f"Successfully created {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: '{chunk['content'][:70].strip()}...'")
    except Exception as e:
        print(f"An error occurred during chunking: {e}")
        print("Could not create or use the chunker")
        sys.exit(1)


    # --- 3. Graph Extraction (Dump Test) ---
    print("\n--- Step 2: Graph Extraction (Dump Test) ---")

    gemini_api_key = "AIzaSyC1EEdzekI_vfPgBLtf_KwPuSy5ahwBJbw"

    if not gemini_api_key:
        print("\n--- Skipping Gemini Test: API key not found. ---")
    else:
        print("\n--- Testing Gemini Client ---")
        try:
            gemini_extractor = GraphExtractorFactory.create_graph_extractor(
                "gemini",
                api_key=gemini_api_key
            )
            for i, chunk in enumerate(chunks):
                print(f"\n--- Processing Chunk {i+1} with Gemini ---")
                gemini_extractor.extract_nodes(chunk['content'])
        except Exception as e:
            print(f"An error occurred creating or running the Gemini extractor: {e}")

    # --- Test Ollama ---
    print("\n--- Testing Ollama Client ---")
    try:
        ollama_extractor = GraphExtractorFactory.create_graph_extractor("ollama")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing Chunk {i+1} with Ollama ---")
            ollama_extractor.extract_nodes(chunk['content'])
    except Exception as e:
        print(f"An error occurred creating or running the Ollama extractor: {e}")
        print("Please ensure the Ollama server is running and accessible at the configured address (default: http://localhost:11434).")


    print("\n--- Test Finished ---")

if __name__ == "__main__":
    main()
