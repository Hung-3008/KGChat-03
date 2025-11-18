"""
UMLS Entity Lookup Module
Search for entities and concepts in the UMLS DuckDB database.
"""

import duckdb
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class UMLSEntity:
    """Represents a UMLS entity"""
    cui: str
    term: str
    language: str
    semantic_types: List[str]
    definitions: List[str]
    related_terms: List[Dict]


class UMLSEntityLookup:
    """Class to lookup entities in the UMLS database"""

    def __init__(self, db_path: str = "data/umls.duckdb"):
        """
        Initialize UMLS lookup

        Args:
            db_path: Path to the DuckDB database
        """
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()

    def _connect(self):
        """Connect to DuckDB once, set read-only mode if available"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
        except TypeError:
            # Older duckdb versions may not accept read_only param
            try:
                self.conn = duckdb.connect(self.db_path)
            except Exception as e:
                raise Exception(f"Cannot connect to UMLS database: {e}")
        except Exception as e:
            raise Exception(f"Cannot connect to UMLS database: {e}")
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def search_entity_by_string(
        self, 
        search_string: str, 
        language: str = "ENG",
        limit: int = 10
    ) -> List[UMLSEntity]:
        """
        Search entities by a string
        
        Args:
            search_string: Search string (exact or partial match)
            language: Language (default: ENG)
            limit: Maximum number of results
            
        Returns:
            List of found UMLSEntity objects
        """
        try:
            # Search in mrconso using LIKE
            query = """
                SELECT DISTINCT c.CUI, c.STR, c.LAT
                FROM mrconso c
                WHERE c.LAT = ? 
                  AND LOWER(c.STR) LIKE LOWER(?)
                LIMIT ?
            """
            
            search_pattern = f"%{search_string}%"
            results = self.conn.execute(query, [language, search_pattern, limit]).fetchall()
            
            entities = []
            for cui, term, lang in results:
                entity = self._build_entity(cui, term, lang)
                if entity:
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def search_exact_match(
        self,
        search_string: str,
        language: str = "ENG"
    ) -> List[UMLSEntity]:
        """
        Search for exact matches
        
        Args:
            search_string: Exact search string
            language: Language (default: ENG)
            
        Returns:
            List of found UMLSEntity objects
        """
        try:
            query = """
                SELECT DISTINCT c.CUI, c.STR, c.LAT
                FROM mrconso c
                WHERE c.LAT = ? 
                  AND LOWER(c.STR) = LOWER(?)
            """
            
            results = self.conn.execute(query, [language, search_string]).fetchall()
            
            entities = []
            for cui, term, lang in results:
                entity = self._build_entity(cui, term, lang)
                if entity:
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"Error during exact match search: {e}")
            return []
    
    def _build_entity(self, cui: str, term: str, language: str) -> Optional[UMLSEntity]:
        """
        Build a UMLSEntity object from a CUI
        
        Args:
            cui: Concept Unique Identifier
            term: Term name
            language: Language
            
        Returns:
            UMLSEntity or None if not found
        """
        try:
            # Get semantic types
            semantic_types = self._get_semantic_types(cui)
            
            # Get definitions
            definitions = self._get_definitions(cui)
            
            # Get related terms
            related_terms = self._get_related_terms(cui)
            
            return UMLSEntity(
                cui=cui,
                term=term,
                language=language,
                semantic_types=semantic_types,
                definitions=definitions,
                related_terms=related_terms
            )
        except Exception as e:
            print(f"Error building entity for {cui}: {e}")
            return None
    
    def _get_semantic_types(self, cui: str) -> List[str]:
        """Get semantic types for a concept"""
        try:
            query = "SELECT STY FROM mrsty WHERE CUI = ?"
            results = self.conn.execute(query, [cui]).fetchall()
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error fetching semantic types for {cui}: {e}")
            return []
    
    def _get_definitions(self, cui: str, limit: int = 5) -> List[str]:
        """Get definitions for a concept"""
        try:
            query = """
                SELECT DEF FROM mrdef 
                WHERE CUI = ? 
                LIMIT ?
            """
            results = self.conn.execute(query, [cui, limit]).fetchall()
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error fetching definitions for {cui}: {e}")
            return []
    
    def _get_related_terms(self, cui: str, limit: int = 10) -> List[Dict]:
        """
        Get related terms for a concept
        
        Returns:
            List of dicts containing: {'related_cui', 'term', 'relation_type'}
        """
        try:
            query = """
                SELECT r.CUI2, c.STR, r.RELA
                FROM mrrel r
                LEFT JOIN mrconso c ON r.CUI2 = c.CUI AND c.LAT = 'ENG' AND c.ISPREF = 'Y'
                WHERE r.CUI1 = ?
                LIMIT ?
            """
            results = self.conn.execute(query, [cui, limit]).fetchall()
            
            related = []
            for related_cui, term, rel_type in results:
                if term:  # Only add if term exists
                    related.append({
                        'related_cui': related_cui,
                        'term': term,
                        'relation_type': rel_type if rel_type else 'unknown'
                    })
            
            return related
        except Exception as e:
            print(f"Error fetching related terms for {cui}: {e}")
            return []
    
    def get_all_terms_for_cui(self, cui: str, language: str = "ENG") -> List[str]:
        """
        Get all terms for a CUI in the specified language
        
        Args:
            cui: Concept Unique Identifier
            language: Language
            
        Returns:
            List of terms
        """
        try:
            query = """
                SELECT DISTINCT STR FROM mrconso
                WHERE CUI = ? AND LAT = ?
                ORDER BY ISPREF DESC
            """
            results = self.conn.execute(query, [cui, language]).fetchall()
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error fetching all terms for {cui}: {e}")
            return []


def find_umls_entities(search_text: str, db_path: str = "data/umls.duckdb", limit: int = 10) -> List[Dict]:
    """
    Utility function to find entities from a text string
    
    Args:
        search_text: Text string to search
        db_path: Path to the UMLS database
        limit: Maximum number of results (default: 10)
        
    Returns:
        List of dicts containing information about found entities
    """
    with UMLSEntityLookup(db_path) as lookup:
        entities = lookup.search_entity_by_string(search_text, limit=limit)
        
        result = []
        for entity in entities:
            result.append({
                'cui': entity.cui,
                'term': entity.term,
                'language': entity.language,
                'semantic_types': entity.semantic_types,
                'definitions': entity.definitions,
                'related_terms': entity.related_terms,
                'all_terms': lookup.get_all_terms_for_cui(entity.cui, entity.language)
            })
        
        return result


def build_umls_prompt_features(
    entities: List[Dict],
    db_path: str = "data/umls.duckdb",
    max_candidates_per_entity: int = 1,
    max_relations_per_pair: int = 2,
) -> Dict:
    """Build compact UMLS features for prompt from a list of entities.

    The result contains:
    - nodes: each entity mapped to at most N UMLS candidates (CUI, term, semantic types)
    - edges: for pairs of entities, UMLS relation patterns between their candidate CUIs
    """
    try:
        from backend.utils.umls_entity_lookup import UMLSEntityLookup
    except ImportError:
        print("Warning: Cannot import UMLSEntityLookup, returning empty UMLS features")
        return {"nodes": [], "edges": []}

    try:
        import duckdb
    except ImportError:
        print("Warning: duckdb is not available, returning empty UMLS features")
        return {"nodes": [], "edges": []}

    # Step 1: choose compact candidates per entity
    umls_nodes: List[Dict] = []
    entity_cui_map: List[Tuple[int, List[str]]] = []

    with UMLSEntityLookup(db_path) as lookup:
        for idx, entity in enumerate(entities):
            name = entity.get("name", "") or ""
            if not name.strip():
                umls_nodes.append({
                    "index": idx,
                    "name": entity.get("name"),
                    "semantic_type": entity.get("semantic_type"),
                    "candidates": [],
                })
                entity_cui_map.append((idx, []))
                continue

            # Prefer exact match, fall back to basic search
            candidates: List[Dict] = []
            cuis_for_entity: List[str] = []

            try:
                exact_results = lookup.search_exact_match(name)
            except Exception:
                exact_results = []

            try:
                basic_results = lookup.search_entity_by_string(name, limit=max_candidates_per_entity)
            except Exception:
                basic_results = []

            ordered_results = list(exact_results) + [r for r in basic_results if r not in exact_results]

            for r in ordered_results[:max_candidates_per_entity]:
                candidates.append({
                    "cui": r.cui,
                    "term": r.term,
                    "semantic_types": r.semantic_types,
                })
                cuis_for_entity.append(r.cui)

            umls_nodes.append({
                "index": idx,
                "name": entity.get("name"),
                "semantic_type": entity.get("semantic_type"),
                "candidates": candidates,
            })
            entity_cui_map.append((idx, cuis_for_entity))

    # Step 2: query UMLS mrrel for relations between candidate CUIs of entity pairs
    if not entity_cui_map:
        return {"nodes": umls_nodes, "edges": []}

    all_cuis = {cui for _, cuis in entity_cui_map for cui in cuis}
    if not all_cuis:
        return {"nodes": umls_nodes, "edges": []}

    con = duckdb.connect(db_path, read_only=True)
    try:
        # Build a temporary table of CUIs to limit the mrrel scan
        cui_list = list(all_cuis)
        con.execute("CREATE OR REPLACE TEMP TABLE tmp_cuis(cui VARCHAR)")
        con.execute("INSERT INTO tmp_cuis VALUES " + ",".join(["(?)" for _ in cui_list]), cui_list)

        rows = con.execute(
            """
            SELECT r.CUI1, r.CUI2, r.REL, r.RELA, r.SAB
            FROM mrrel r
            JOIN tmp_cuis t1 ON r.CUI1 = t1.cui
            JOIN tmp_cuis t2 ON r.CUI2 = t2.cui
            """
        ).fetchall()
    finally:
        con.close()

    # Index relations by CUI pair for fast lookup
    rel_map: Dict[Tuple[str, str], List[Dict]] = {}
    for cui1, cui2, rel, rela, sab in rows:
        key = (cui1, cui2)
        rel_list = rel_map.setdefault(key, [])
        if len(rel_list) < max_relations_per_pair:
            rel_list.append({
                "rel": rel,
                "rela": rela,
                "sab": sab,
            })

    umls_edges: List[Dict] = []

    # For each entity pair, collect relation patterns between their candidate CUIs
    n = len(entity_cui_map)
    for i in range(n):
        idx_i, cuis_i = entity_cui_map[i]
        for j in range(i + 1, n):
            idx_j, cuis_j = entity_cui_map[j]
            pair_relations: List[Dict] = []

            for cui_i in cuis_i:
                for cui_j in cuis_j:
                    for key in ((cui_i, cui_j), (cui_j, cui_i)):
                        if key in rel_map:
                            for r in rel_map[key]:
                                if len(pair_relations) >= max_relations_per_pair:
                                    break
                                pair_relations.append({
                                    "source_cui": key[0],
                                    "target_cui": key[1],
                                    "rel": r["rel"],
                                    "rela": r["rela"],
                                    "sab": r["sab"],
                                })
                    if len(pair_relations) >= max_relations_per_pair:
                        break
                if len(pair_relations) >= max_relations_per_pair:
                    break

            if pair_relations:
                umls_edges.append({
                    "source_index": idx_i,
                    "target_index": idx_j,
                    "relations": pair_relations,
                })

    return {"nodes": umls_nodes, "edges": umls_edges}

def look_up(entities: List[Dict], db_path: str = "data/umls.duckdb") -> List[Dict]:
    """
    Search for UMLS candidates for each entity.
    
    Args:
        entities: List of entities to search
        db_path: Path to UMLS database
        
    Returns:
        List of entities with 'extra_information' field containing candidates
    """
    result_entities = []
    
    with UMLSEntityLookup(db_path) as lookup:
        for entity in entities:
            entity_copy = dict(entity)
            entity_name = entity.get("name", "")
            
            if not entity_name or not entity_name.strip():
                entity_copy["extra_information"] = {"candidates": []}
                result_entities.append(entity_copy)
                continue
            
            candidates = {
                "basic_search": [],
                "exact_search": []
            }
            
            try:
                basic_results = lookup.search_entity_by_string(entity_name, limit=3)
                for result in basic_results:
                    candidates["basic_search"].append({
                        "cui": result.cui,
                        "term": result.term,
                        "semantic_types": result.semantic_types,
                        "definitions": result.definitions[:1] if result.definitions else []
                    })
            except Exception as e:
                print(f"Error in basic search for '{entity_name}': {e}")
            
            try:
                exact_results = lookup.search_exact_match(entity_name)
                for result in exact_results[:3]:
                    candidates["exact_search"].append({
                        "cui": result.cui,
                        "term": result.term,
                        "semantic_types": result.semantic_types,
                        "definitions": result.definitions[:1] if result.definitions else []
                    })
            except Exception as e:
                print(f"Error in exact search for '{entity_name}': {e}")
            
            entity_copy["extra_information"] = candidates
            result_entities.append(entity_copy)
    
    return result_entities

if __name__ == "__main__":
    # Example usage
    import json
    
    # Search for an entity
    search_term = "diabetes"
    print(f"\n=== Search: '{search_term}' ===\n")
    
    results = find_umls_entities(search_term)
    
    if results:
        for i, entity in enumerate(results[:3], 1):  # Display first 3 results
            print(f"Result {i}:")
            print(f"  CUI: {entity['cui']}")
            print(f"  Term: {entity['term']}")
            print(f"  Semantic Types: {', '.join(entity['semantic_types'])}")
            
            if entity['definitions']:
                print(f"  Definition: {entity['definitions'][0][:100]}...")
            
            if entity['related_terms']:
                print(f"  Related terms (top 3):")
                for rel in entity['related_terms'][:3]:
                    print(f"    - {rel['term']} ({rel['relation_type']})")
            
            if entity['all_terms']:
                print(f"  All terms: {', '.join(entity['all_terms'][:5])}")
            
            print()
    else:
        print("No results found")
