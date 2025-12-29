import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field

# Add project root to sys.path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from IHDGTPipeline import IHDGTPipeline, ConfigManager
except ImportError:
    # If running from scripts dir, IHDGTPipeline.py is in parent
    sys.path.append(str(Path(__file__).parent.parent))
    from IHDGTPipeline import IHDGTPipeline, ConfigManager
from backend.llm.llm_factory import LLMFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("medqa_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("medqa_eval")

def load_medqa_data(file_path: str):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def format_query(item: dict) -> str:
    # Format: Question \n Options: A. ... B. ...
    q = item['question']
    options = item.get('options', {})
    opt_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
    return f"{q}\n\nOptions:\n{opt_str}"

def load_processed_ids(output_path: str) -> set:
    processed = set()
    if not os.path.exists(output_path):
        return processed
    
    with open(output_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "question_id" in data:
                    processed.add(data["question_id"])
            except json.JSONDecodeError:
                continue
    return processed

class EvaluationResponse(BaseModel):
    is_correct: int = Field(..., description="1 if the answer is correct (matches correct option), 0 otherwise.")
    reasoning: str = Field(..., description="Brief reasoning for the evaluation verdict.")

def evaluate_answer(llm_client, question: str, options: dict, correct_idx: str, generated_text: str) -> int:
    try:
        prompt = f"""
        You are an evaluator. Determine if the Generated Answer matches the Correct Option for the given Question.
        
        Question: {question}
        Options: {json.dumps(options)}
        Correct Option: {correct_idx} ({options.get(correct_idx, '')})
        
        Generated Answer: 
        {generated_text}
        
        Task: verification.
        Does the Generated Answer explicitly or implicitly choose the Correct Option?
        """
        
        # Use structured generation
        response = llm_client.generate(prompt, format=EvaluationResponse)
        
        if isinstance(response, EvaluationResponse):
            return response.is_correct
        elif isinstance(response, dict):
             return int(response.get("is_correct", 0))
        else:
             # Fallback manual parse if needed, but generate with format should return obj or dict
             # Depending on implementation of LLMClient
             if hasattr(response, "is_correct"):
                 return int(response.is_correct)
             return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Evaluate IHDGTPipeline on MedQA")
    parser.add_argument("--data_path", type=str, 
                        default="test_data/MedQA/data_clean/questions/US/test.jsonl",
                        help="Path to MedQA jsonl file")
    parser.add_argument("--config", type=str, 
                        default="backend/configs/ihdpgt_config.yml",
                        help="Path to pipeline config")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit number of samples to process")
    parser.add_argument("--output", type=str, default="results/medqa_evaluation_results.jsonl",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Ensure paths are absolute or relative to project root
    data_path = Path(project_root) / args.data_path if not os.path.isabs(args.data_path) else Path(args.data_path)
    config_path = Path(project_root) / args.config if not os.path.isabs(args.config) else Path(args.config)
    output_path = Path(project_root) / args.output if not os.path.isabs(args.output) else Path(args.output)
    
    # Create output dir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {data_path}...")
    dataset = load_medqa_data(str(data_path))
    
    if args.limit:
        dataset = dataset[:args.limit]
        logger.info(f"Limiting to first {args.limit} samples.")
    
    logger.info(f"Initializing pipeline with {config_path}...")
    logger.info(f"Initializing pipeline with {config_path}...")
    pipeline = IHDGTPipeline(str(config_path))
    
    # Initialize Evaluation LLM (reuse same config)
    config = ConfigManager.load_config(str(config_path))
    eval_llm = LLMFactory.create_client(config.get("Retrieval", {}).get("LLM", {}))
    
    logger.info("Starting evaluation...")
    
    # Check for existing progress
    processed_ids = load_processed_ids(str(output_path))
    if processed_ids:
        logger.info(f"files exist. Found {len(processed_ids)} processed questions. Resuming...")

    # Open output file in append mode (resume)
    with open(output_path, 'a') as f_out:
        for i, item in enumerate(tqdm(dataset)):
            if i in processed_ids:
                continue
                
            try:
                query = format_query(item)
                logger.info(f"Processing Q{i+1}: {item['question'][:50]}...")
                
                # Run Pipeline
                generated_answer = pipeline.run(query)
                
                result_entry = {
                    "question_id": i,
                    "question": item['question'],
                    "options": item['options'],
                    "correct_answer": item['answer'],
                    "correct_answer_idx": item['answer_idx'],
                    "generated_text": generated_answer
                }
                
                # Evaluate correctness
                is_correct = evaluate_answer(eval_llm, item['question'], item['options'], item['answer_idx'], generated_answer)
                result_entry["is_correct"] = is_correct
                logger.info(f"  Is Correct: {is_correct}")
                
                f_out.write(json.dumps(result_entry) + "\n")
                f_out.flush() 
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                f_out.write(json.dumps({"question_id": i, "error": str(e)}) + "\n")
    
    pipeline.cleanup()
    logger.info(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
