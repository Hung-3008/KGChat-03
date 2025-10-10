import os
import json
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from backend.pipeline.graph_extraction.extractor_factory import GraphExtractorFactory
from data.test.nodes_extraction.preprocessing import load_dataset
from evaluation.logger import get_logger, save_results
from evaluation.metrics import evaluate, evaluate_entity_names
import time
from collections import defaultdict
from tqdm import tqdm

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
    if not isinstance(dataset_names, list):
        logger.error("The 'datasets' parameter in the config file must be a list.")
        return
    logger.info(f"Found {len(dataset_names)} dataset(s) to run in config: {', '.join(dataset_names)}")
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
            # Load Node Prompts
            with open(os.path.join(base_prompt_path, prompt_settings["node_system_prompt"]), 'r') as f:
                node_system_prompt = f.read()
            with open(os.path.join(base_prompt_path, prompt_settings["node_user_prompt"]), 'r') as f:
                node_user_prompt_template = f.read()
            
            
            prompt_data = {
                "node_system_prompt": node_system_prompt, 
                "node_user_prompt_template": node_user_prompt_template,
                "mode": prompt_settings.get("mode", "standard"),
                "k": prompt_settings.get("k", 1)
            }
            if prompt_data["mode"] == "interactive":
                with open(os.path.join(base_prompt_path, prompt_settings["feedback_prompt"]), 'r') as f:
                    prompt_data["node_feedback_prompt_template"] = f.read()

            prompts['default'] = prompt_data
        except (IOError, KeyError) as e:
            logger.error(f"Could not load prompt files. Check paths in config.yaml. Error: {e}")
    
    results = {}
    prompt_summary = defaultdict(lambda: {
        "total_precision": 0, "total_recall": 0, "total_f1": 0, 
        "total_sklearn_precision": 0, "total_sklearn_recall": 0, "total_sklearn_f1": 0, 
        "count": 0
    })

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
                
                samples_to_process = dataset
                if max_samples is not None:
                    samples_to_process = samples_to_process[:max_samples]

                pbar = tqdm(enumerate(samples_to_process), total=len(samples_to_process), desc=f"Model: {model_name}, Dataset: {dataset_name}")
                for i, (text, ground_truth) in pbar:
                    try:
                        predicted_nodes = extractor.extract_nodes(
                            document_content=text,
                            system_prompt=prompt_data["node_system_prompt"],
                            user_prompt_template=prompt_data["node_user_prompt_template"],
                            feedback_prompt_template=prompt_data.get("node_feedback_prompt_template"),
                            mode=prompt_data["mode"],
                            k=prompt_data["k"]
                        )
                        predicted_entities = [node.properties for node in predicted_nodes]

                        evaluation_metrics = evaluate(predicted_entities, ground_truth["entities"], fuzzy_threshold=fuzzy_threshold)

                        # New evaluation on entity names
                        predicted_entity_names = [p['text'].lower() for p in predicted_entities]
                        ground_truth_entity_names = [gt['text'].lower() for gt in ground_truth["entities"]]
                        sklearn_metrics = evaluate_entity_names(predicted_entity_names, ground_truth_entity_names)

                        # Add to summary data (using fuzzy scores)
                        fuzzy_scores = evaluation_metrics.get("scores", {}).get("fuzzy", {})
                        prompt_summary[prompt_name]["total_precision"] += fuzzy_scores.get("precision", 0)
                        prompt_summary[prompt_name]["total_recall"] += fuzzy_scores.get("recall", 0)
                        prompt_summary[prompt_name]["total_f1"] += fuzzy_scores.get("f1", 0)

                        prompt_summary[prompt_name]["total_sklearn_precision"] += sklearn_metrics.get("precision", 0)
                        prompt_summary[prompt_name]["total_sklearn_recall"] += sklearn_metrics.get("recall", 0)
                        prompt_summary[prompt_name]["total_sklearn_f1"] += sklearn_metrics.get("f1", 0)

                        prompt_summary[prompt_name]["count"] += 1

                        results[model_name][dataset_name][prompt_name].append({
                            "sample_id": i,
                            "input_text": text,
                            "predicted_entities": predicted_entities,
                            "ground_truth": ground_truth["entities"],
                            "evaluation": evaluation_metrics,
                            "sklearn_evaluation": sklearn_metrics
                        })

                    except Exception as e:
                        #logger.error(f"Error during generation for sample {i}: {e}")
                        results[model_name][dataset_name][prompt_name].append({"sample_id": i, "error": str(e)})
                    
                    if time_between_requests > 0:
                        time.sleep(time_between_requests)

    # --- Calculate Averages and Final Report ---
    summary_report = defaultdict(dict)
    for prompt_name, summary_data in prompt_summary.items():
        count = summary_data["count"]
        if count > 0:
            # Fuzzy Match
            avg_fuzzy_precision = summary_data["total_precision"] / count
            avg_fuzzy_recall = summary_data["total_recall"] / count
            avg_fuzzy_f1 = summary_data["total_f1"] / count
            
            # Sklearn Exact Match
            avg_sklearn_precision = summary_data["total_sklearn_precision"] / count
            avg_sklearn_recall = summary_data["total_sklearn_recall"] / count
            avg_sklearn_f1 = summary_data["total_sklearn_f1"] / count

            summary_report[prompt_name] = {
                "samples_processed": count,
                "fuzzy_match_avg": {
                    "precision": avg_fuzzy_precision,
                    "recall": avg_fuzzy_recall,
                    "f1": avg_fuzzy_f1
                },
                "sklearn_exact_match_avg": {
                    "precision": avg_sklearn_precision,
                    "recall": avg_sklearn_recall,
                    "f1": avg_sklearn_f1
                }
            }

    final_report = {
        "config": config,
        "summary": summary_report,
        "results": results
    }
    save_results(final_report)
    
    # --- Log Summary to Console ---
    logger.info("\n--- Experiment Summary ---")
    for prompt_name, summary in summary_report.items():
        logger.info(f"Prompt: {prompt_name} ({summary['samples_processed']} samples)")
        logger.info(f"  Average Fuzzy Precision: {summary['fuzzy_match_avg']['precision']:.4f}")
        logger.info(f"  Average Fuzzy Recall:    {summary['fuzzy_match_avg']['recall']:.4f}")
        logger.info(f"  Average Fuzzy F1-Score:  {summary['fuzzy_match_avg']['f1']:.4f}")
        logger.info(f"  Average Sklearn Precision: {summary['sklearn_exact_match_avg']['precision']:.4f}")
        logger.info(f"  Average Sklearn Recall:    {summary['sklearn_exact_match_avg']['recall']:.4f}")
        logger.info(f"  Average Sklearn F1-Score:  {summary['sklearn_exact_match_avg']['f1']:.4f}")
    logger.info("------------------------\n")

    logger.info("Experiment finished.")
    return final_report
