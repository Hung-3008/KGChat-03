import os
import csv
import ast

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
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_LOADERS[dataset_name]()
