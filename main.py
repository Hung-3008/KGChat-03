"""
Medical Entity Extraction Pipeline:
1. Load PMC sample data from data/500_samples_pmc
2. Load configuration from backend/configs/configs.yml
3. Chunk documents using section-based chunking
4. Extract medical entities with embeddings using LLM and transformer encoder
"""
import yaml
import json
from pathlib import Path
from backend.llm.llm_factory import LLMFactory
from backend.chunkers.section_chunking import SectionChunker
from backend.encoders.transformer_encoder import TransformerEncoder
from backend.node_extractor.node_extractor import NodeExtractor


def main(data_dir: str = "data/500_samples_pmc", config_path: str = "backend/configs/configs.yml", n_samples: int = 5):
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        configs = yaml.safe_load(f) or {}
    llm_config = configs.get("LLM", {})
    encoder_config = configs.get("Encoder", {})
    llm_client = LLMFactory.create_client(llm_config)
    encoder = TransformerEncoder(
        model_name=encoder_config.get("model_name", "NeuML/pubmedbert-base-embeddings"),
        device=encoder_config.get("device", "cpu")
    )
    extractor = NodeExtractor(
        llm_client=llm_client,
        model_name=llm_config.get("model", "llama3.1:8b"),
        embedding_model=encoder_config.get("model_name", "NeuML/pubmedbert-base-embeddings"),
        encoder=encoder,
        device=encoder_config.get("device", "cpu")
    )
    chunker = SectionChunker()
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob("*.json"))[:n_samples]
    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                document = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue
        chunks = chunker.chunk(document)
        print(f"\n{'='*80}")
        print(f"File: {json_file.name}")
        print(f"Title: {document.get('title', 'N/A')}")
        print(f"Chunks: {len(chunks)}")
        print(f"{'='*80}")
        for chunk_idx, chunk in enumerate(chunks, 1):
            entities = extractor.extract(chunk)
            if entities:
                print(f"\nChunk {chunk_idx} - Extracted {len(entities)} entities")
                for entity in entities[:5]:
                    print(f"  - {entity['name']}: {entity['mention'][:100]}...")
            # save enities to a JSON file
            output_base_path = "output/entities_extracted"
            output_base = Path(output_base_path)
            output_base.mkdir(parents=True, exist_ok=True)
            output_file = output_base / f"{json_file.stem}_chunk{chunk_idx}_entities.json"
            try:
                with output_file.open("w", encoding="utf-8") as out_f:
                    json.dump(entities, out_f, ensure_ascii=False, indent=2)
                print(f"  -> Saved entities to {output_file}")
            except Exception as e:
                print(f"  -> Error saving entities to {output_file}: {e}")


if __name__ == "__main__":
    main(n_samples=2)

