import os
import csv
import ast
import json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_ncbi_csv_dataset(file_path: str):
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return []

    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entities = []
            try:
                starts = ast.literal_eval(row["start"])
                ends = ast.literal_eval(row["end"])
                mentions = ast.literal_eval(row["mention"])
                types = ast.literal_eval(row["type"])
                semantic_ids = ast.literal_eval(row["semantic_id"])
            except (ValueError, SyntaxError):
                continue

            for i in range(len(starts)):
                entities.append({
                    "text": mentions[i],
                    "type": types[i],
                    "start": int(starts[i]),
                    "end": int(ends[i]),
                    "semantic_id": semantic_ids[i]
                })
            
            text = f"{row['title']} {row['abstract']}"
            dataset.append((text, {"entities": entities}))

    return dataset

def _load_bc5dr_csv_dataset(file_path: str):
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return []

    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entities = []
            try:
                labels = ast.literal_eval(row["labels"])
            except (ValueError, SyntaxError):
                continue

            for label in labels:
                if len(label) == 5:
                    start, end, mention, entity_type, semantic_id = label
                    entities.append({
                        "text": mention,
                        "type": entity_type,
                        "start": int(start),
                        "end": int(end),
                        "semantic_id": semantic_id
                    })

            doc_text = f"{row['title']} {row['text']}"
            dataset.append((doc_text, {"entities": entities}))

    return dataset

def _load_ncbi_corpus():
    file_path = os.path.join(DATA_DIR, "NCBItestset_corpus.csv")
    return _load_ncbi_csv_dataset(file_path)

def _load_bc5dr_corpus():
    file_path = os.path.join(DATA_DIR, "BC5DR.csv")
    return _load_bc5dr_csv_dataset(file_path)


DATASET_LOADERS = {
    "NCBItestset_corpus": _load_ncbi_corpus,
    "bc5dr": _load_bc5dr_corpus,
}

def load_dataset(dataset_name: str):
    # First, support known dataset keys
    if dataset_name in DATASET_LOADERS:
        return DATASET_LOADERS[dataset_name]()

    # Next, support a filesystem path (directory of JSON files or a single CSV/JSON file)
    if os.path.exists(dataset_name):
        # If a directory, attempt to load JSON files (e.g., PMC result JSONs)
        if os.path.isdir(dataset_name):
            dataset = []
            for fname in sorted(os.listdir(dataset_name)):
                if not fname.lower().endswith('.json'):
                    continue
                file_path = os.path.join(dataset_name, fname)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                except Exception:
                    # skip unreadable files
                    continue

                # Collect text from content_sections if available
                content_sections = doc.get('content_sections') or doc.get('content', [])
                if isinstance(content_sections, list):
                    text = "\n\n".join([s.get('content', '') for s in content_sections if isinstance(s, dict)])
                else:
                    # fallback to raw text fields
                    text = doc.get('text') or doc.get('abstract') or ''

                dataset.append((text, doc.get('metadata', {})))

            return dataset

        # If a single file, support CSVs handled above or JSON
        if os.path.isfile(dataset_name):
            if dataset_name.lower().endswith('.json'):
                try:
                    with open(dataset_name, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                except Exception:
                    raise ValueError(f"Failed to read JSON dataset: {dataset_name}")

                content_sections = doc.get('content_sections') or doc.get('content', [])
                if isinstance(content_sections, list):
                    text = "\n\n".join([s.get('content', '') for s in content_sections if isinstance(s, dict)])
                else:
                    text = doc.get('text') or doc.get('abstract') or ''

                return [(text, doc.get('metadata', {}))]

    raise ValueError(f"Unknown dataset: {dataset_name}")
