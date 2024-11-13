"""Graph data preprocessing utilities."""

import numpy as np
from typing import Optional, List, Dict, Tuple
from core import Transformer
import networkx as nx

class GraphFeatureExtractor(Transformer):
    """Extract features from graph structures."""
    
    def __init__(self, features: List[str] = ['degree', 'centrality', 'clustering']):
        self.features = features
        
    def fit(self, X: List[nx.Graph], y: Optional[np.ndarray] = None) -> 'GraphFeatureExtractor':
        """Fit extractor (no-op)."""
        return self
        
    def transform(self, X: List[nx.Graph]) -> np.ndarray:
        """Extract graph features."""
        features_list = []
        
        for graph in X:
            features_dict = {}
            
            if 'degree' in self.features:
                degrees = list(dict(graph.degree()).values())
                features_dict['avg_degree'] = np.mean(degrees)
                features_dict['max_degree'] = np.max(degrees)
                
            if 'centrality' in self.features:
                centrality = nx.betweenness_centrality(graph)
                features_dict['avg_centrality'] = np.mean(list(centrality.values()))
                
            if 'clustering' in self.features:
                clustering = nx.clustering(graph)
                features_dict['avg_clustering'] = np.mean(list(clustering.values()))
                
            features_list.append(np.array(list(features_dict.values())))
            
        return np.array(features_list)

class GraphNormalizer(Transformer):
    """Normalize graph adjacency matrices."""
    
    def __init__(self, normalization: str = 'symmetric'):
        self.normalization = normalization
        
    def fit(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> 'GraphNormalizer':
        """Fit normalizer (no-op)."""
        return self
        
    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize adjacency matrices."""
        normalized = []
        
        for adj in X:
            if self.normalization == 'symmetric':
                # A' = D^(-1/2) A D^(-1/2)
                D = np.diag(np.sum(adj, axis=1))
                D_inv_sqrt = np.linalg.inv(np.sqrt(D))
                normalized.append(D_inv_sqrt @ adj @ D_inv_sqrt)
            elif self.normalization == 'random_walk':
                # A' = D^(-1) A
                D = np.diag(np.sum(adj, axis=1))
                D_inv = np.linalg.inv(D)
                normalized.append(D_inv @ adj)
                
        return normalized 