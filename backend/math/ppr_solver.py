
import networkx as nx
from typing import List, Dict

class PPRSolver:
    def __init__(self, damping: float = 0.85):
        self.damping = damping

    def run_ppr(self, nodes: List[Dict], relationships: List[Dict], seeds: List[str]) -> Dict[str, float]:
        """
        Runs Personalized PageRank on the provided subgraph.
        """
        if not nodes:
            return {}
            
        # Build Graph
        G = nx.Graph() # Undirected or Directed? Usually knowledge graphs are directed but influence flows both ways in some models. 
                       # Let's use DiGraph but maybe allow reverse flow? 
                       # Standard PageRank on KGs usually assumes directed or adds reverse edges.
                       # For simplicity and robustness, let's use DiGraph.
        
        G = nx.DiGraph()
        
        for n in nodes:
            G.add_node(n['id'], **n)
            
        for r in relationships:
            G.add_edge(r['source'], r['target'], type=r['type'], weight=r.get('weight', 1.0))
            
        # Personalization vector
        # Distribute weight evenly among seeds that exist in the graph
        valid_seeds = [s for s in seeds if s in G]
        if not valid_seeds:
            return {}
            
        seed_weight = 1.0 / len(valid_seeds)
        personalization = {node: (seed_weight if node in valid_seeds else 0.0) for node in G.nodes()}
        
        try:
            ppr_scores = nx.pagerank(
                G,
                alpha=self.damping,
                personalization=personalization,
                weight='weight'  # Use edge weights if available
            )
            return ppr_scores
        except Exception as e:
            print(f"PPR Calculation failed: {e}") 
            # Fallback? Return equal scores or zeros?
            return {}
