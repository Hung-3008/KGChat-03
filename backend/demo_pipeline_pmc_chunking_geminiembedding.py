import json
import logging
import asyncio
from typing import List, Dict, Any
from backend.pipeline.chunking import DocumentChunker
from backend.llm.providers.gemini.gemini_client import GeminiClient
from backend.llm.providers.gemini.gemini_config import GeminiConfig
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_12")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pmc_json(file_path: str) -> Dict[str, Any]:
    """Load PMC JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_full_text_from_pmc(pmc_data: Dict[str, Any]) -> str:
    """Extract full text from PMC data."""
    full_text = ""

    title = pmc_data.get('title', '')
    if title:
        full_text += f"# {title}\n\n"

    metadata = pmc_data.get('metadata', {})
    if metadata:
        full_text += "## Document Information\n"
        full_text += f"- Word count: {metadata.get('word_count', 'N/A')}\n"
        full_text += f"- Estimated reading time: {metadata.get('estimated_reading_time_minutes', 'N/A')} minutes\n\n"

    content_sections = pmc_data.get('content_sections', [])
    for section in content_sections:
        header = section.get('header', '')
        content = section.get('content', '')
        if header and content:
            full_text += f"## {header}\n\n"
            full_text += f"{content}\n\n"

    return full_text.strip()


async def demonstrate_complete_pipeline(pmc_file: str, use_real_api: bool = False):
    """Demo pipeline pmc - embedding - vector storage."""

    logger.info("DEMO: COMPLETE PMC → VECTOR PIPELINE")
    logger.info("=" * 70)

    # Step 1: Load PMC data
    logger.info("\nStep 1: Loading PMC data...")
    pmc_data = load_pmc_json(pmc_file)
    full_text = extract_full_text_from_pmc(pmc_data)
    logger.info(f"Loaded: {pmc_data.get('title', 'Unknown')}")
    logger.info(f"Text length: {len(full_text):,} characters")

    # Step 2: Section-based Chunking
    logger.info("\nStep 2: Section-based chunking...")
    chunker = DocumentChunker(
        max_chunk_tokens=2000,  # ~8000 characters _ 1500 words
        overlap_tokens=150,     # ~600 characters overlap _ 112 words
        min_chunk_tokens=100    # Minimum chunk size
    )

    chunk_metadata = {
        "document_id": pmc_file,
        "document_title": pmc_data.get('title', 'Unknown'),
        "source": "PMC",
        "knowledge_level": 1
    }

    logger.info("Chunker config:")
    logger.info(f"   - Max chunk tokens: {chunker.max_chunk_tokens}")
    logger.info(f"   - Overlap tokens: {chunker.overlap_tokens}")
    logger.info(f"   - Min chunk tokens: {chunker.min_chunk_tokens}")

    # Sử dụng section-based chunking thay vì text-based chunking
    chunks = chunker.create_section_chunks(pmc_data, chunk_metadata)
    logger.info(f"Created {len(chunks)} section-based chunks")

    # Step 2.5: Analyze chunks
    logger.info("\nStep 2.5: Analyzing chunks...")
    total_chars = sum(len(chunk['content']) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0

    logger.info(f"   - Total chunks: {len(chunks)}")
    logger.info(f"   - Average chunk size: {avg_chunk_size:.0f} characters")
    logger.info(f"   - Total characters in chunks: {total_chars:,}")
    logger.info(f"   - Coverage ratio: {total_chars / len(full_text):.2%}")

    # Step 3: Initialize Gemini
    logger.info("\nStep 3: Initializing Gemini client...")

    if use_real_api:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_12")
        logger.info("Using real Gemini API (if API key is valid)")
    else:
        # Sử dụng fallback
        api_key = "demo_key"
        logger.info("Using fallback vectors (demo mode)")

    gemini_config = GeminiConfig(
        api_key=api_key,
        embedding_model="models/text-embedding-004"
    )
    gemini_client = GeminiClient(gemini_config)
    logger.info("Gemini client initialized")

    # Step 4: Show sample chunks
    logger.info("\nStep 4: Sample chunks...")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        section_type = chunk['metadata'].get('section_type', 'unknown')
        section_header = chunk['metadata'].get('section_header', 'Unknown')

        logger.info(f"\n--- Chunk {i+1} ---")
        logger.info(f"ID: {chunk['metadata']['chunk_id']}")
        logger.info(f"Index: {chunk['metadata']['chunk_index']}")
        logger.info(f"Section: {section_header} ({section_type})")
        logger.info(f"Tokens: {chunk.get('tokens', 'N/A')}")
        logger.info(f"Content preview: {chunk['content'][:200]}...")

    # Step 5: Embedding process
    logger.info("\nStep 5: Embedding process...")

    embedding_results = []
    successful_embeddings = 0
    failed_embeddings = 0

    for i, chunk in enumerate(chunks):
        section_type = chunk['metadata'].get('section_type', 'unknown')
        section_header = chunk['metadata'].get('section_header', 'Unknown')

        logger.info(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
        logger.info(f"Section: {section_header} ({section_type})")
        logger.info(f"Content: {chunk['content'][:100]}...")

        try:
            # Generate embedding
            embedding = await gemini_client.embed_single(chunk['content'])

            # Create result
            result = {
                "chunk_id": chunk['metadata']['chunk_id'],
                "content": chunk['content'],
                "metadata": chunk['metadata'],
                "embedding_vector": embedding,
                "embedding_dimensions": len(embedding),
                "embedding_model": gemini_config.embedding_model,
                "success": True
            }

            embedding_results.append(result)
            successful_embeddings += 1

            logger.info(f"Success! Dimensions: {len(embedding)}")

        except Exception as e:
            logger.error(f"Failed: {str(e)}")

            # Fallback result
            fallback_vector = [0.0] * 768
            result = {
                "chunk_id": chunk['metadata']['chunk_id'],
                "content": chunk['content'],
                "metadata": chunk['metadata'],
                "embedding_vector": fallback_vector,
                "embedding_dimensions": 768,
                "embedding_model": "fallback_zero_vector",
                "success": False,
                "error": str(e)
            }

            embedding_results.append(result)
            failed_embeddings += 1

    # Step 6: Analysis
    logger.info("\nStep 6: Analysis...")

    total_chars = sum(len(result['content']) for result in embedding_results)
    avg_chunk_size = total_chars / \
        len(embedding_results) if embedding_results else 0

    logger.info("Pipeline Results:")
    logger.info(f"   - Total chunks: {len(embedding_results)}")
    logger.info(f"   - Successful embeddings: {successful_embeddings}")
    logger.info(f"   - Failed embeddings: {failed_embeddings}")
    logger.info(f"   - Total characters: {total_chars:,}")
    logger.info(f"   - Average chunk size: {avg_chunk_size:.0f} characters")

    if successful_embeddings > 0:
        sample_vector = next(r['embedding_vector']
                             for r in embedding_results if r['success'])
        logger.info(f"   - Vector dimensions: {len(sample_vector)}")
        logger.info(f"   - Sample values: {sample_vector[:5]}")

    # Step 7: Prepare for vector storage
    logger.info("\nStep 7: Preparing for vector storage...")

    # prepare for vector storage (qdrant)
    vector_points = []

    for i, result in enumerate(embedding_results):
        point = {
            "id": result['chunk_id'],
            "vector": result['embedding_vector'],
            "payload": {
                "content": result['content'],
                "metadata": result['metadata'],
                "embedding_model": result['embedding_model'],
                "success": result['success']
            }
        }
        vector_points.append(point)

    logger.info(f"Prepared {len(vector_points)} points for vector storage")

    # Step 8: Save results
    logger.info("\nStep 8: Saving results...")

    output_data = {
        "pipeline_info": {
            "document_title": pmc_data.get('title', 'Unknown'),
            "source": "PMC",
            "total_chunks": len(chunks),
            "processed_chunks": len(embedding_results),
            "successful_embeddings": successful_embeddings,
            "failed_embeddings": failed_embeddings
        },
        "chunking_config": {
            "max_chunk_tokens": chunker.max_chunk_tokens,
            "overlap_tokens": chunker.overlap_tokens,
            "min_chunk_tokens": chunker.min_chunk_tokens
        },
        "embedding_config": {
            "model": gemini_config.embedding_model,
            "dimensions": 768,
            "api_used": successful_embeddings > 0
        },
        "vector_points": vector_points,
        "chunks_summary": []
    }

    # create summary
    for result in embedding_results[:5]:
        chunk_summary = {
            "chunk_id": result['chunk_id'],
            "section_type": result['metadata'].get('section_type', 'unknown'),
            "section_header": result['metadata'].get('section_header', 'Unknown'),
            "content_preview": result['content'][:200] + "...",
            "metadata": result['metadata'],
            "embedding_info": {
                "dimensions": result['embedding_dimensions'],
                "model": result['embedding_model'],
                "success": result['success']
            }
        }
        output_data["chunks_summary"].append(chunk_summary)

    # save results
    output_file = f"/Users/maitiendung/TAI LIEU/HOME/research/KGCHAT_new_ver/output/pmc_chunking_gemini_embedding/pmc_pipeline_results_{pmc_file.replace('.json', '')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_file}")

    # Step 9: Next steps
    logger.info("\nStep 9: Next steps...")
    logger.info("What you can do next:")
    logger.info("1. Store vectors in Qdrant database")
    logger.info("2. Implement similarity search")
    logger.info("3. Build knowledge graph from chunks")
    logger.info("4. Create RAG system for Q&A")
    logger.info("5. Analyze chunk relationships")

    # Show chunking strategies
    show_chunking_strategies()

    return embedding_results


def show_chunking_strategies():

    logger.info("\n\nCHUNKING STRATEGIES")
    logger.info("=" * 70)

    strategies = [
        {
            "name": "Fixed-size with overlap",
            "config": {"max_tokens": 1000, "overlap": 200},
            "pros": "Consistent size, context preservation",
            "cons": "May split sentences",
            "use_case": "General documents"
        },
        {
            "name": "Sentence-based",
            "config": {"sentences_per_chunk": 10},
            "pros": "Preserves sentence integrity",
            "cons": "Variable chunk sizes",
            "use_case": "Academic papers"
        },
        {
            "name": "Paragraph-based",
            "config": {"paragraphs_per_chunk": 2},
            "pros": "Preserves logical units",
            "cons": "Very variable sizes",
            "use_case": "Structured documents"
        },
        {
            "name": "Section-based (Current)",
            "config": {"each_header": "1_chunk"},
            "pros": "Preserves semantic structure, perfect for research papers",
            "cons": "Variable sizes, requires structured input",
            "use_case": "PMC papers, structured documents"
        },
        {
            "name": "Semantic chunking",
            "config": {"similarity_threshold": 0.8},
            "pros": "Semantically coherent chunks",
            "cons": "Computationally expensive",
            "use_case": "High-quality retrieval"
        }
    ]

    for strategy in strategies:
        logger.info(f"\n- {strategy['name']}")
        logger.info(f"   Config: {strategy['config']}")
        logger.info(f"   Pros: {strategy['pros']}")
        logger.info(f"   Cons: {strategy['cons']}")
        logger.info(f"   Best for: {strategy['use_case']}")


def main():
    """Main function."""
    pmc_file = "PMC2749957_result.json"

    # use real api gemini
    use_real_api = True

    logger.info(
        f"\nRunning in {'Real API' if use_real_api else 'Demo'} mode...")

    # Run pipeline
    results = asyncio.run(
        demonstrate_complete_pipeline(pmc_file, use_real_api))

    logger.info(f"\nPipeline completed! Processed {len(results)} chunks.")

    # Summary
    logger.info("\n\nFINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("1. PMC data loaded and processed")
    logger.info("2. Section-based chunking applied (each header = 1 chunk)")
    logger.info("3. Chunks prepared for embedding")
    logger.info("4. Embedding vectors generated (fallback)")
    logger.info("5. Results analyzed and saved")

    logger.info("\nKey Points:")
    logger.info("- Section-based chunking: each header becomes a separate chunk")
    logger.info("- Preserves semantic structure of documents")
    logger.info("- Metadata includes section type and header information")
    logger.info("- Better for structured documents like research papers")


if __name__ == "__main__":
    main()
