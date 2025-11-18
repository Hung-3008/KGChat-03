from __future__ import annotations
from typing import Dict, List, Optional, Tuple

CLUSTER_DEFINITIONS: Dict[Optional[str], List[Tuple[str, str, Optional[str]]]] = {
    "activity": [
        ("T052", "Activity", None),
        ("T053", "Behavior", "T052"),
        ("T054", "Social_Behavior", "T053"),
        ("T055", "Individual_Behavior", "T053"),
        ("T056", "Daily_or_Recreational_Activity", "T052"),
        ("T057", "Occupational_Activity", "T052"),
        ("T058", "Health_Care_Activity", "T052"),
        ("T059", "Laboratory_Procedure", "T058"),
        ("T060", "Diagnostic_Procedure", "T058"),
        ("T061", "Therapeutic_or_Preventive_Procedure", "T058"),
        ("T062", "Research_Activity", "T052"),
        ("T063", "Molecular_Biology_Research_Technique", "T062"),
        ("T064", "Governmental_or_Regulatory_Activity", "T052"),
        ("T065", "Educational_Activity", "T052"),
        ("T066", "Machine_Activity", "T052"),
    ],
    "phenomenon": [
        ("T067", "Phenomenon_or_Process", None),
        ("T037", "Injury_or_Poisoning", "T067"),
        ("T068", "Human_caused_Phenomenon_or_Process", "T067"),
        ("T069", "Environmental_Effect_of_Humans", "T067"),
        ("T070", "Natural_Phenomenon_or_Process", "T067"),
        ("T038", "Biologic_Function", "T067"),
        ("T039", "Physiologic_Function", "T038"),
        ("T040", "Organism_Function", "T039"),
        ("T041", "Mental_Process", "T067"),
        ("T042", "Organ_or_Tissue_Function", "T067"),
        ("T043", "Cell_Function", "T042"),
        ("T044", "Molecular_Function", "T043"),
        ("T045", "Genetic_Function", "T044"),
        ("T049", "Cell_or_Molecular_Dysfunction", "T044"),
        ("T046", "Pathologic_Function", "T067"),
        ("T047", "Disease_or_Syndrome", "T046"),
        ("T048", "Mental_or_Behavioral_Dysfunction", "T046"),
        ("T191", "Neoplastic_Process", "T046"),
        ("T050", "Experimental_Model_of_Disease", "T046"),
    ],
    "physical_object": [
        ("T072", "Physical_Object", None),
        ("T001", "Organism", "T072"),
        ("T005", "Virus", "T001"),
        ("T007", "Bacterium", "T001"),
        ("T194", "Archaeon", "T001"),
        ("T204", "Eukaryote", "T001"),
        ("T002", "Plant", "T204"),
        ("T004", "Fungus", "T204"),
        ("T008", "Animal", "T204"),
        ("T010", "Vertebrate", "T008"),
        ("T011", "Amphibian", "T010"),
        ("T012", "Bird", "T010"),
        ("T013", "Fish", "T010"),
        ("T014", "Reptile", "T010"),
        ("T015", "Mammal", "T010"),
        ("T016", "Human", "T015"),
        ("T017", "Anatomical_Structure", "T072"),
        ("T018", "Embryonic_Structure", "T017"),
        ("T021", "Fully_Formed_Anatomical_Structure", "T017"),
        ("T023", "Body_Part_Organ_or_Organ_Component", "T021"),
        ("T024", "Tissue", "T021"),
        ("T025", "Cell", "T021"),
        ("T026", "Cell_Component", "T021"),
        ("T028", "Gene_or_Genome", "T021"),
        ("T190", "Anatomical_Abnormality", "T017"),
        ("T019", "Congenital_Abnormality", "T190"),
        ("T020", "Acquired_Abnormality", "T190"),
        ("T073", "Manufactured_Object", "T072"),
        ("T074", "Medical_Device", "T073"),
        ("T203", "Drug_Delivery_Device", "T074"),
        ("T075", "Research_Device", "T073"),
        ("T200", "Clinical_Drug", "T073"),
        ("T167", "Substance", "T072"),
        ("T031", "Body_Substance", "T167"),
        ("T103", "Chemical", "T167"),
        ("T104", "Chemical_Viewed_Structurally", "T103"),
        ("T109", "Organic_Chemical", "T104"),
        ("T114", "Nucleic_Acid_Nucleoside_or_Nucleotide", "T109"),
        ("T116", "Amino_Acid_Peptide_or_Protein", "T109"),
        ("T196", "Element_Ion_or_Isotope", "T104"),
        ("T197", "Inorganic_Chemical", "T104"),
        ("T120", "Chemical_Viewed_Functionally", "T103"),
        ("T121", "Pharmacologic_Substance", "T120"),
        ("T195", "Antibiotic", "T121"),
        ("T122", "Biomedical_or_Dental_Material", "T120"),
        ("T123", "Biologically_Active_Substance", "T120"),
        ("T125", "Hormone", "T123"),
        ("T126", "Enzyme", "T123"),
        ("T127", "Vitamin", "T123"),
        ("T129", "Immunologic_Factor", "T123"),
        ("T192", "Receptor", "T123"),
        ("T130", "Indicator_Reagent_or_Diagnostic_Aid", "T120"),
        ("T131", "Hazardous_or_Poisonous_Substance", "T120"),
        ("T168", "Food", "T167"),
    ],
    "conceptual_entity": [
        ("T077", "Conceptual_Entity", None),
        ("T032", "Organism_Attribute", "T077"),
        ("T201", "Clinical_Attribute", "T032"),
        ("T033", "Finding", "T077"),
        ("T034", "Laboratory_or_Test_Result", "T033"),
        ("T184", "Sign_or_Symptom", "T033"),
        ("T078", "Idea_or_Concept", "T077"),
        ("T079", "Temporal_Concept", "T078"),
        ("T080", "Qualitative_Concept", "T078"),
        ("T081", "Quantitative_Concept", "T078"),
        ("T082", "Spatial_Concept", "T078"),
        ("T029", "Body_Location_or_Region", "T082"),
        ("T030", "Body_Space_or_Junction", "T082"),
        ("T083", "Geographic_Area", "T082"),
        ("T085", "Molecular_Sequence", "T082"),
        ("T086", "Nucleotide_Sequence", "T085"),
        ("T087", "Amino_Acid_Sequence", "T085"),
        ("T088", "Carbohydrate_Sequence", "T085"),
        ("T169", "Functional_Concept", "T078"),
        ("T022", "Body_System", "T169"),
        ("T090", "Occupation_or_Discipline", "T077"),
        ("T091", "Biomedical_Occupation_or_Discipline", "T090"),
        ("T092", "Organization", "T077"),
        ("T093", "Health_Care_Related_Organization", "T092"),
        ("T094", "Professional_Society", "T092"),
        ("T095", "Self_help_or_Relief_Organization", "T092"),
        ("T096", "Group", "T077"),
        ("T097", "Professional_or_Occupational_Group", "T096"),
        ("T098", "Population_Group", "T096"),
        ("T099", "Family_Group", "T096"),
        ("T100", "Age_Group", "T096"),
        ("T101", "Patient_or_Disabled_Group", "T096"),
        ("T102", "Group_Attribute", "T077"),
        ("T170", "Intellectual_Product", "T077"),
        ("T089", "Regulation_or_Law", "T170"),
        ("T185", "Classification", "T170"),
        ("T171", "Language", "T077"),
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


