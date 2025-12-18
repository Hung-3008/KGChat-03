from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# Optimized CLUSTER_DEFINITIONS matching reduced semantic types
# Reduced from 113 total types to 44 types (61% reduction)

CLUSTER_DEFINITIONS: Dict[Optional[str], List[Tuple[str, str, Optional[str]]]] = {
    "activity": [
        # Reduced from 15 to 9 types - removed non-medical activities
        ("T052", "Activity", None),
        ("T053", "Behavior", "T052"),
        ("T058", "Health_Care_Activity", "T052"),
        ("T059", "Laboratory_Procedure", "T058"),
        ("T060", "Diagnostic_Procedure", "T058"),
        ("T061", "Therapeutic_or_Preventive_Procedure", "T058"),
        ("T062", "Research_Activity", "T052"),
        ("T063", "Molecular_Biology_Research_Technique", "T062"),
        ("T064", "Governmental_or_Regulatory_Activity", "T052"),
    ],
    "phenomenon": [
        # Reduced from 19 to 11 types - kept core disease/pathology types
        ("T047", "Disease_or_Syndrome", None),
        ("T191", "Neoplastic_Process", None),
        ("T037", "Injury_or_Poisoning", None),
        ("T046", "Pathologic_Function", None),
        ("T048", "Mental_or_Behavioral_Dysfunction", "T046"),
        ("T184", "Sign_or_Symptom", None),
        ("T039", "Physiologic_Function", None),
        ("T042", "Organ_or_Tissue_Function", None),
        ("T049", "Cell_or_Molecular_Dysfunction", None),
        ("T038", "Biologic_Function", None),
        ("T041", "Mental_Process", None),
    ],
    "physical_object": [
        # Reduced from 55 to 16 types - removed organism taxonomy, kept medical entities
        ("T023", "Body_Part_Organ_or_Organ_Component", None),
        ("T024", "Tissue", None),
        ("T025", "Cell", None),
        ("T028", "Gene_or_Genome", None),
        ("T190", "Anatomical_Abnormality", None),
        ("T019", "Congenital_Abnormality", "T190"),
        ("T020", "Acquired_Abnormality", "T190"),
        ("T200", "Clinical_Drug", None),
        ("T121", "Pharmacologic_Substance", None),
        ("T195", "Antibiotic", "T121"),
        ("T125", "Hormone", None),
        ("T126", "Enzyme", None),
        ("T127", "Vitamin", None),
        ("T074", "Medical_Device", None),
        ("T031", "Body_Substance", None),
        ("T001", "Organism", None),  # Only for disease-causing organisms
    ],
    "conceptual_entity": [
        # Reduced from 38 to 8 types - removed organizations, groups, occupations
        ("T033", "Finding", None),
        ("T184", "Sign_or_Symptom", None),
        ("T034", "Laboratory_or_Test_Result", "T033"),
        ("T201", "Clinical_Attribute", None),
        ("T029", "Body_Location_or_Region", None),
        ("T022", "Body_System", None),
        ("T081", "Quantitative_Concept", None),
        ("T079", "Temporal_Concept", None),
    ],
}


def build_hierarchy_tree(cluster_definitions: Dict[Optional[str], List[Tuple[str, str, Optional[str]]]]) -> Dict[str, Dict[str, int]]:
    """Build a mapping of semantic type name to its depth for each cluster."""
    hierarchy_tree: Dict[str, Dict[str, int]] = {
        k: {} for k in cluster_definitions.keys() if k is not None
    }

    for cluster_name, semantic_types in cluster_definitions.items():
        if cluster_name is None:
            continue

        id_to_name: Dict[str, str] = {}
        id_to_parent: Dict[str, Optional[str]] = {}

        for type_id, type_name, parent_id in semantic_types:
            id_to_name[type_id] = type_name
            id_to_parent[type_id] = parent_id

        for type_id, type_name, _ in semantic_types:
            depth = 0
            current_parent_id = id_to_parent.get(type_id)

            while current_parent_id is not None and current_parent_id in id_to_parent:
                depth += 1
                current_parent_id = id_to_parent.get(current_parent_id)

            hierarchy_tree[cluster_name][type_name] = depth

    return hierarchy_tree


def normalize_entity(entity: str) -> str:
    """Normalize an entity string for comparison."""
    return entity.strip().lower()


def filter_entities_by_hierarchy(
    extracted_entities: Dict[str, Dict[str, List[str]]],
    hierarchy_tree: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Keep, for each entity, the semantic type(s) with the maximum depth.
    If multiple types share the same maximum depth, return all of them.
    """
    entity_mapping: Dict[str, List[Dict]] = {}

    for cluster_name, cluster_data in extracted_entities.items():
        for semantic_type, entities in cluster_data.items():
            depth = hierarchy_tree.get(cluster_name, {}).get(semantic_type, 0)
            for entity in entities:
                normalized = normalize_entity(entity)
                entity_mapping.setdefault(normalized, []).append({
                    "original_entity": entity,
                    "semantic_type": semantic_type,
                    "cluster": cluster_name,
                    "depth": depth
                })

    best_semantic_types: Dict[str, Optional[Dict]] = {}
    for ent, types in entity_mapping.items():
        if len(types) == 1:
            best_semantic_types[ent] = types[0]
        else:
            sorted_types = sorted(types, key=lambda x: x["depth"], reverse=True)
            max_depth = sorted_types[0]["depth"]
            max_depth_types = [t for t in sorted_types if t["depth"] == max_depth]
            best_semantic_types[ent] = max_depth_types[0] if len(max_depth_types) == 1 else None

    filtered_entities: Dict[str, Dict[str, List[str]]] = {c: {} for c in extracted_entities.keys()}

    for ent, best in best_semantic_types.items():
        if best is None:
            max_depth = max(t["depth"] for t in entity_mapping[ent])
            max_depth_types = [t for t in entity_mapping[ent] if t["depth"] == max_depth]
            for type_info in max_depth_types:
                cluster = type_info["cluster"]
                semantic_type = type_info["semantic_type"]
                original_entity = type_info["original_entity"]
                filtered_entities.setdefault(cluster, {}).setdefault(semantic_type, [])
                if original_entity not in filtered_entities[cluster][semantic_type]:
                    filtered_entities[cluster][semantic_type].append(original_entity)
        else:
            cluster = best["cluster"]
            semantic_type = best["semantic_type"]
            original_entity = best["original_entity"]
            filtered_entities.setdefault(cluster, {}).setdefault(semantic_type, [])
            if original_entity not in filtered_entities[cluster][semantic_type]:
                filtered_entities[cluster][semantic_type].append(original_entity)

    return filtered_entities
