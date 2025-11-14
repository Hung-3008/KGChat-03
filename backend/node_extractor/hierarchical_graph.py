"""Hierarchical graph for UMLS semantic types filtering."""

from typing import Dict, List


class HierarchicalGraph:
    """UMLS semantic type hierarchy graph for filtering entities."""
    
    def __init__(self):
        self.field_to_depth: Dict[str, int] = self._build_field_depths()
    
    def _build_field_depths(self) -> Dict[str, int]:
        """Build field_name -> depth mapping from UMLS hierarchy.
        
        Based on 4 cluster hierarchies from output/umls/clusters/:
        - T052_Activity.json (15 nodes, depth 1-4)
        - T067_Phenomenon_or_Process.json (19 nodes, depth 1-6)
        - T072_Physical_Object.json (54 nodes, depth 1-7)
        - T077_Conceptual_Entity.json (37 nodes, depth 1-5)
        """
        return {
            # T052 Activity Cluster
            "activity": 1,
            "behavior": 2,
            "social_behavior": 3,
            "individual_behavior": 3,
            "daily_or_recreational_activity": 2,
            "occupational_activity": 2,
            "health_care_activity": 3,
            "laboratory_procedure": 4,
            "diagnostic_procedure": 4,
            "therapeutic_or_preventive_procedure": 4,
            "research_activity": 3,
            "molecular_biology_research_technique": 4,
            "governmental_or_regulatory_activity": 3,
            "educational_activity": 3,
            "machine_activity": 2,
            
            # T067 Phenomenon_or_Process Cluster
            "phenomenon_or_process": 1,
            "injury_or_poisoning": 2,
            "biologic_function": 3,
            "physiologic_function": 4,
            "organism_function": 5,
            "mental_process": 6,
            "organ_or_tissue_function": 5,
            "cell_function": 5,
            "molecular_function": 5,
            "genetic_function": 6,
            "pathologic_function": 4,
            "disease_or_syndrome": 5,
            "mental_or_behavioral_dysfunction": 6,
            "cell_or_molecular_dysfunction": 5,
            "experimental_model_of_disease": 5,
            "human_caused_phenomenon_or_process": 2,
            "environmental_effect_of_humans": 3,
            "natural_phenomenon_or_process": 2,
            "neoplastic_process": 6,
            
            # T072 Physical_Object Cluster
            "physical_object": 1,
            "organism": 2,
            "plant": 4,
            "fungus": 4,
            "virus": 3,
            "bacterium": 3,
            "animal": 4,
            "vertebrate": 5,
            "amphibian": 6,
            "bird": 6,
            "fish": 6,
            "reptile": 6,
            "mammal": 6,
            "human": 7,
            "anatomical_structure": 2,
            "embryonic_structure": 3,
            "congenital_abnormality": 4,
            "acquired_abnormality": 4,
            "fully_formed_anatomical_structure": 3,
            "body_part_organ_or_organ_component": 4,
            "tissue": 4,
            "cell": 4,
            "cell_component": 4,
            "gene_or_genome": 4,
            "body_substance": 3,
            "manufactured_object": 2,
            "medical_device": 3,
            "research_device": 3,
            "chemical": 3,
            "chemical_viewed_structurally": 4,
            "organic_chemical": 5,
            "nucleic_acid_nucleoside_or_nucleotide": 6,
            "amino_acid_peptide_or_protein": 6,
            "chemical_viewed_functionally": 4,
            "pharmacologic_substance": 5,
            "biomedical_or_dental_material": 5,
            "biologically_active_substance": 5,
            "hormone": 6,
            "enzyme": 6,
            "vitamin": 6,
            "immunologic_factor": 6,
            "indicator_reagent_or_diagnostic_aid": 5,
            "hazardous_or_poisonous_substance": 5,
            "substance": 2,
            "food": 3,
            "anatomical_abnormality": 3,
            "receptor": 6,
            "archaeon": 3,
            "antibiotic": 6,
            "element_ion_or_isotope": 5,
            "inorganic_chemical": 5,
            "clinical_drug": 3,
            "drug_delivery_device": 4,
            "eukaryote": 3,
            
            # T077 Conceptual_Entity Cluster
            "conceptual_entity": 1,
            "body_system": 4,
            "body_location_or_region": 4,
            "body_space_or_junction": 4,
            "organism_attribute": 2,
            "finding": 2,
            "laboratory_or_test_result": 3,
            "idea_or_concept": 2,
            "temporal_concept": 3,
            "qualitative_concept": 3,
            "quantitative_concept": 3,
            "spatial_concept": 3,
            "geographic_area": 4,
            "molecular_sequence": 4,
            "nucleotide_sequence": 5,
            "amino_acid_sequence": 5,
            "carbohydrate_sequence": 5,
            "regulation_or_law": 3,
            "occupation_or_discipline": 2,
            "biomedical_occupation_or_discipline": 3,
            "organization": 2,
            "health_care_related_organization": 3,
            "professional_society": 3,
            "self_help_or_relief_organization": 3,
            "group": 2,
            "professional_or_occupational_group": 3,
            "population_group": 3,
            "family_group": 3,
            "age_group": 3,
            "patient_or_disabled_group": 3,
            "group_attribute": 2,
            "functional_concept": 3,
            "intellectual_product": 2,
            "language": 2,
            "sign_or_symptom": 3,
            "classification": 3,
            "clinical_attribute": 3,
        }
    
    def get_depth(self, field_name: str) -> int:
        """Get depth of a semantic type field."""
        return self.field_to_depth.get(field_name, 0)
    
    def filter_by_depth(self, entities_by_field: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Filter entities prioritizing deepest semantic types.
        
        For each unique entity appearing in multiple fields:
        - Keep only in the field with greatest depth (most specific)
        - Remove from other fields
        
        Args:
            entities_by_field: Dict mapping field names to entity lists
            
        Returns:
            Filtered dict with entities only in their deepest field
        """
        # Track entity -> [(field, depth), ...] mapping
        entity_to_fields: Dict[str, List[tuple]] = {}
        
        for field, entities in entities_by_field.items():
            if not entities:
                continue
            depth = self.get_depth(field)
            for entity in entities:
                if entity not in entity_to_fields:
                    entity_to_fields[entity] = []
                entity_to_fields[entity].append((field, depth))
        
        # Build filtered output: keep only in deepest field
        filtered: Dict[str, List[str]] = {field: [] for field in entities_by_field.keys()}
        
        for entity, field_depths in entity_to_fields.items():
            if len(field_depths) == 1:
                # Single field - keep as is
                field, _ = field_depths[0]
                filtered[field].append(entity)
            else:
                # Multiple fields - keep only in deepest
                deepest_field = max(field_depths, key=lambda x: x[1])[0]
                filtered[deepest_field].append(entity)
        
        return filtered


# Global singleton
_hierarchy = None


def get_hierarchy() -> HierarchicalGraph:
    """Get singleton hierarchy instance."""
    global _hierarchy
    if _hierarchy is None:
        _hierarchy = HierarchicalGraph()
    return _hierarchy
