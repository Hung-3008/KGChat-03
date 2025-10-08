import os
import json
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from backend.pipeline.graph_extraction.extractor_factory import GraphExtractorFactory
from data.test.nodes_extraction.preprocessing import load_dataset
from evaluation.logger import get_logger, save_results
from evaluation.metrics import evaluate
import time
from collections import defaultdict

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

    # --- Load Settings ---
    exp_config = config.get("graph_extraction", {})
    model_names = exp_config.get("models", [])
    dataset_names = exp_config.get("datasets", [])
    provider = exp_config.get("provider")
    max_samples = exp_config.get("max_samples", None)
    fuzzy_threshold = exp_config.get("fuzzy_threshold", 0.85)
    time_between_requests = exp_config.get("time_between_requests", 0)

    # --- Load Prompts ---
    prompts = {}
    prompt_settings = exp_config.get("prompt_settings", {})
    if prompt_settings:
        base_prompt_path = prompt_settings.get("base_path", "")
        try:
            with open(os.path.join(base_prompt_path, prompt_settings["node_system_prompt"]), 'r') as f:
                system_prompt = f.read()
            with open(os.path.join(base_prompt_path, prompt_settings["node_user_prompt"]), 'r') as f:
                user_prompt_template = f.read()
            
            prompt_data = {
                "system_prompt": system_prompt, 
                "user_prompt_template": user_prompt_template,
                "mode": prompt_settings.get("mode", "standard"),
                "k": prompt_settings.get("k", 1)
            }
            if prompt_data["mode"] == "interactive":
                with open(os.path.join(base_prompt_path, prompt_settings["feedback_prompt"]), 'r') as f:
                    prompt_data["feedback_prompt_template"] = f.read()

            prompts['default'] = prompt_data
        except (IOError, KeyError) as e:
            logger.error(f"Could not load prompt files. Check paths in config.yaml. Error: {e}")
    
    results = {}
    prompt_summary = defaultdict(lambda: {"total_precision": 0, "total_recall": 0, "total_f1": 0, "count": 0})

    for model_name in model_names:
        logger.info(f"Loading model: {model_name}")
        try:
            extractor = GraphExtractorFactory.create_graph_extractor(
                provider,
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

            for prompt_name, prompt_data in prompts.items():
                logger.info(f"--- Running with prompt: {prompt_name} ---")
                results[model_name][dataset_name][prompt_name] = []
                
                num_samples_to_process = len(dataset)
                if max_samples is not None:
                    num_samples_to_process = min(len(dataset), max_samples)

                for i, (text, ground_truth) in enumerate(dataset):
                    if max_samples is not None and i >= max_samples:
                        break

                    logger.info(f"Processing sample {i+1}/{num_samples_to_process}")
                    
                    try:
                        predicted_nodes = extractor.extract_nodes(
                            document_content=text, **prompt_data
                        )
                        predicted_entities = [node.properties for node in predicted_nodes]

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
                            "predicted_entities": predicted_entities,
                            "ground_truth": ground_truth["entities"],
                            "evaluation": evaluation_metrics
                        })

                    except Exception as e:
                        logger.error(f"Error during generation for sample {i}: {e}")
                        results[model_name][dataset_name][prompt_name].append({"sample_id": i, "error": str(e)})
                    
                    if time_between_requests > 0:
                        time.sleep(time_between_requests)

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
