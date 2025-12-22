prompt_evaluate_relation = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
A: 1. {{language.human_language.main_country (Score: 0.4)}}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {{language.human_language.countries_spoken_in (Score: 0.3)}}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {{base.rosetta.languoid.parent (Score: 0.2)}}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Q: {query}
Relations: {relations}
 """

prompt_score_entity = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1), Do not use the entities that already exist in the Old Entities list.
Q: What medications are commonly prescribed for Type 2 diabetes?
Relation: disease.medications
Entities: Metformin; Insulin; Sulfonylureas; DPP-4 inhibitors; GLP-1 receptor agonists; SGLT2 inhibitors
A: 1. {{Metformin (Score: 0.3)}}: Metformin is the most commonly prescribed first-line medication.
2. {{Insulin (Score: 0.2)}}: Insulin is often prescribed when other medications are not enough.
3. {{Sulfonylureas (Score: 0.2)}}: Sulfonylureas are another common class of oral medications.
4. {{DPP-4 inhibitors (Score: 0.1)}}: Used as add-on therapy.
5. {{GLP-1 receptor agonists (Score: 0.1)}}: Injectable medications.
6. {{SGLT2 inhibitors (Score: 0.1)}}: Newer class of oral medications.

Q: {query}
Relation: {relations}
Old Entities: {old_entities}
Entites: {entities}"""

# Danh gia xem co du thong tin de tra loi cau hoi khong
prompt_evaluate = """Given a question and the associated retrieved information (including entity descriptions and knowledge graph triplets), you are asked to answer whether it's sufficient for you to answer the question with this information and your knowledge (Yes or No).

Q: What are the recommended dietary changes for managing Type 2 diabetes?
Retrieved Information: 
Type 2 Diabetes, DISEASE, A chronic condition that affects the way the body processes blood sugar (glucose).
Type 2 Diabetes, disease.dietary_recommendations, Reduce carbohydrate intake
Type 2 Diabetes, disease.dietary_recommendations, Increase fiber consumption
Type 2 Diabetes, disease.dietary_recommendations, Limit processed foods
A: {{Yes}}. The given information provides sufficient details about the recommended dietary changes for managing Type 2 diabetes, including reducing carbohydrate intake, increasing fiber consumption, and limiting processed foods.

Q: What are the long-term complications of Type 2 diabetes?
Retrieved Information: 
Type 2 Diabetes, DISEASE, A chronic condition that affects the way the body processes blood sugar (glucose).
Type 2 Diabetes, disease.complications, Heart disease
Type 2 Diabetes, disease.complications, Kidney damage
A: {{No}}. While the given information mentions some complications (heart disease and kidney damage), it doesn't provide a comprehensive list of all possible long-term complications of Type 2 diabetes, such as nerve damage, eye problems, and foot complications.

Q: {query}
Retrieved Information: 
{triplets}"""