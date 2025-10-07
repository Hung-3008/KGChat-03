import os
import json
import yaml
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from backend.pipeline.graph_extraction.extractor_factory import GraphExtractorFactory
from data.test.nodes_extraction.preprocessing import load_dataset
from evaluation.logger import get_logger, save_results
from evaluation.metrics import evaluate
import time

logger = get_logger(__name__, log_file="logs/experiment.log")

def run_experiment(config_path: str = "test/nodes_extraction/config.yaml"):

    logger.info("Starting experiment...")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return

    experiment_settings = config.get("graph_extraction", {})
   
    
    # Extract settings from config
    model_names = experiment_settings.get("models", [])
    dataset_names = experiment_settings.get("datasets", [])
    provider = experiment_settings.get("provider")
    max_samples = experiment_settings.get("max_samples", None)
    fuzzy_threshold = experiment_settings.get("fuzzy_threshold", 0.85)
    
    time_sleep = 0
    if provider == "gemini":
        time_sleep = 3
        logger.info(f"Provider {provider} identified as Gemini model.")
    else:
        logger.info(f"Provider {provider} identified as Ollama model.")

    results = {}
    prompt_summary = defaultdict(lambda: {"total_precision": 0, "total_recall": 0, "total_f1": 0, "count": 0})

    for model_name in model_names:
        logger.info(f"Loading model: {model_name}")
        try:
            extractor = GraphExtractorFactory.create_graph_extractor(
                provider, 
                config_file_path=config_path, 
                model_name=model_name,
            )
            results[model_name] = {}
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue

        for dataset_name in dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            if not dataset:
                logger.warning(f"Dataset {dataset_name} is empty or could not be loaded. Skipping.")
                continue
            results[model_name][dataset_name] = {}

            prompt_name = "default_prompt"
            results[model_name][dataset_name][prompt_name] = []
            
            num_samples_to_process = len(dataset)
            if max_samples is not None:
                num_samples_to_process = min(len(dataset), max_samples)

            for i, (text, ground_truth) in enumerate(dataset):
                
                time.sleep(time_sleep)  # Sleep to avoid rate limits for Gemini models      
                if max_samples is not None and i >= max_samples:
                    break

                logger.info(f"Processing sample {i+1}/{num_samples_to_process}")
                
                try:
                    predicted_nodes = extractor.extract_nodes(text)
                    predicted_entities = [node.properties for node in predicted_nodes]
                    output = json.dumps({"entities": predicted_entities}) # Create a dummy output for logging

                    evaluation_metrics = evaluate(predicted_entities, ground_truth["entities"], fuzzy_threshold=fuzzy_threshold)

                    # Add to summary data (using fuzzy scores)
                    fuzzy_scores = evaluation_metrics.get("scores", {}).get("fuzzy", {})
                    prompt_summary[prompt_name]["total_precision"] += fuzzy_scores.get("precision", 0)
                    prompt_summary[prompt_name]["total_recall"] += fuzzy_scores.get("recall", 0)
                    prompt_summary[prompt_name]["total_f1"] += fuzzy_scores.get("f1", 0)
                    prompt_summary[prompt_name]["count"] += 1

                    results[model_name][dataset_name][prompt_name].append({
                        "sample_id": i,
                        "input_text": text,
                        "raw_output": output,
                        "predicted_entities": predicted_entities,
                        "ground_truth": ground_truth["entities"],
                        "evaluation": evaluation_metrics
                    })

                except Exception as e:
                    logger.error(f"Error during generation for sample {i}: {e}")
                    results[model_name][dataset_name][prompt_name].append({
                        "sample_id": i,
                        "error": str(e)
                    })

    save_results(results)
    
    logger.info("\n--- Experiment Summary (Fuzzy Match) ---")
    for prompt_name, summary_data in prompt_summary.items():
        count = summary_data["count"]
        if count > 0:
            avg_precision = summary_data["total_precision"] / count
            avg_recall = summary_data["total_recall"] / count
            avg_f1 = summary_data["total_f1"] / count
            logger.info(f"Prompt: {prompt_name}")
            logger.info(f"  Average Fuzzy Precision: {avg_precision:.4f}")
            logger.info(f"  Average Fuzzy Recall:    {avg_recall:.4f}")
            logger.info(f"  Average Fuzzy F1-Score:  {avg_f1:.4f}")
    logger.info("------------------------\n")

    logger.info("Experiment finished.")
    return results
