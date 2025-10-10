import jellyfish
from scipy.optimize import linear_sum_assignment
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def _deduplicate_by_text(entities: list[dict]) -> list[dict]:
    """Removes duplicate entities based on the 'text' field only."""
    seen = set()
    deduplicated = []
    for entity in entities:
        key = entity['text']
        if key not in seen:
            seen.add(key)
            deduplicated.append(entity)
    return deduplicated

def calculate_metrics(tp, fp, fn):
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def fuzzy_match_score(s1: str, s2: str) -> float:
    """Computes a combined fuzzy match score between two strings."""
    if not s1 or not s2:
        return 0.0
    jaro_dist = jellyfish.jaro_winkler_similarity(s1, s2)
    lev_sim = 1 - (jellyfish.levenshtein_distance(s1, s2) / max(len(s1), len(s2)))
    return max(jaro_dist, lev_sim)

def evaluate(predicted_entities: list[dict], ground_truth_entities: list[dict], fuzzy_threshold: float = 1.0):
    """Implements a simplified NER evaluation plan based on lowercase text matching."""

    # 1. Lowercase all entity text, as per user request.
    lower_preds = [{'text': e['text'].lower(), 'type': e['description']} for e in predicted_entities]
    lower_truths = [{'text': gt['text'].lower(), 'type': gt['type']} for gt in ground_truth_entities]

    # 2. Deduplicate based on the lowercased text.
    pred_dedup = _deduplicate_by_text(lower_preds)
    truth_dedup = _deduplicate_by_text(lower_truths)

    # --- Matching Process ---
    unmatched_preds = list(range(len(pred_dedup)))
    unmatched_truths = list(range(len(truth_dedup)))
    matches = {"exact": [], "fuzzy": []}
    errors = defaultdict(list)

    # Pass 1: Exact Case-Insensitive Matching
    for i in list(unmatched_preds):
        for j in list(unmatched_truths):
            if pred_dedup[i]['text'] == truth_dedup[j]['text']:
                matches['exact'].append((i, j))
                unmatched_preds.remove(i)
                unmatched_truths.remove(j)
                break

    # Pass 2: Fuzzy Matching on remaining items
    if unmatched_preds and unmatched_truths:
        cost_matrix = np.zeros((len(unmatched_preds), len(unmatched_truths)))
        for i, pred_idx in enumerate(unmatched_preds):
            for j, truth_idx in enumerate(unmatched_truths):
                score = fuzzy_match_score(pred_dedup[pred_idx]['text'], truth_dedup[truth_idx]['text'])
                cost_matrix[i, j] = 1 - score  # Use cost for minimization

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            score = 1 - cost_matrix[r, c]
            if score >= fuzzy_threshold:
                pred_idx = unmatched_preds[r]
                truth_idx = unmatched_truths[c]
                matches['fuzzy'].append((pred_idx, truth_idx))
        
        matched_preds_in_fuzzy = {unmatched_preds[r] for r, c in zip(row_ind, col_ind) if 1 - cost_matrix[r, c] >= fuzzy_threshold}
        matched_truths_in_fuzzy = {unmatched_truths[c] for r, c in zip(row_ind, col_ind) if 1 - cost_matrix[r, c] >= fuzzy_threshold}
        unmatched_preds = [p for p in unmatched_preds if p not in matched_preds_in_fuzzy]
        unmatched_truths = [t for t in unmatched_truths if t not in matched_truths_in_fuzzy]

    # --- Calculate TP, FP, FN and Metrics ---
    tp_exact = len(matches['exact'])
    # "normalized" score is now the same as "exact" because we only lowercase.
    tp_normalized = tp_exact 
    tp_fuzzy = tp_exact + len(matches['fuzzy'])

    fp = len(unmatched_preds)
    fn = len(unmatched_truths)

    scores = {
        "exact": calculate_metrics(tp_exact, fp + len(matches['fuzzy']), fn + len(matches['fuzzy'])),
        "normalized": calculate_metrics(tp_normalized, fp + len(matches['fuzzy']), fn + len(matches['fuzzy'])),
        "fuzzy": calculate_metrics(tp_fuzzy, fp, fn)
    }

    # --- Error Analysis ---
    errors['boundary_or_other_fp'] = [pred_dedup[i] for i in unmatched_preds]
    errors['boundary_or_other_fn'] = [truth_dedup[j] for j in unmatched_truths]

    # --- Final Output ---
    return {
        "fuzzy_threshold": fuzzy_threshold,
        "scores": scores,
        "errors": {
            "boundary_fp": len(errors['boundary_or_other_fp']),
            "boundary_fn": len(errors['boundary_or_other_fn'])
        }
    }

def evaluate_entity_names(predicted_entity_names: list[str], ground_truth_entity_names: list[str]):
    """Calculates precision, recall, and F1-score for lists of entity names using scikit-learn."""
    
    pred_set = set(predicted_entity_names)
    truth_set = set(ground_truth_entity_names)
    
    all_labels = sorted(list(pred_set.union(truth_set)))
    
    if not all_labels:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 0}
        
    y_pred = [1 if label in pred_set else 0 for label in all_labels]
    y_true = [1 if label in truth_set else 0 for label in all_labels]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }
