"""Graph data preprocessing utilities."""

import numpy as np
from typing import Optional, List, Dict, Tuple
from core import Transformer
import networkx as nx
from core import EPSILON, ValidationError

class GraphFeatureExtractor(Transformer):
    """Extract features from graph structures."""
    
    def __init__(self, features: List[str] = ['degree', 'centrality', 'clustering']):
        """Initialize feature extractor.
        
        Args:
            features: List of features to extract ('degree', 'centrality', 'clustering')
        """
        if not isinstance(features, list) or not features:
            raise ValueError("features must be a non-empty list")
            
        valid_features = {'degree', 'centrality', 'clustering'}
        invalid_features = set(features) - valid_features
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}")
            
        self.features = features
        
    def fit(self, X: List[nx.Graph], y: Optional[np.ndarray] = None) -> 'GraphFeatureExtractor':
        """Fit extractor (no-op)."""
        return self
        
    def transform(self, X: List[nx.Graph]) -> np.ndarray:
        """Extract graph features.
        
        Args:
            X: List of networkx graphs
            
        Returns:
            Array of shape (n_graphs, n_features) containing extracted features
            
        Raises:
            ValidationError: If input graphs are empty or invalid
        """
        if not X:
            raise ValidationError("Input graph list cannot be empty")
            
        features_list = []
        
        for graph in X:
            if not isinstance(graph, nx.Graph):
                raise ValidationError(f"Expected nx.Graph, got {type(graph)}")
                
            if len(graph) == 0:
                raise ValidationError("Empty graphs are not supported")
                
            features_dict = {}
            
            if 'degree' in self.features:
                degrees = [d for _, d in nx.degree(graph)]
                features_dict['avg_degree'] = np.mean(degrees)
                features_dict['max_degree'] = np.max(degrees)
                
            if 'centrality' in self.features:
                try:
                    centrality = nx.betweenness_centrality(graph)
                    features_dict['avg_centrality'] = np.mean(list(centrality.values()))
                except:
                    # Handle disconnected graphs
                    features_dict['avg_centrality'] = 0.0
                
            if 'clustering' in self.features:
                clustering_coeffs = nx.clustering(graph)
                if isinstance(clustering_coeffs, dict):
                    features_dict['avg_clustering'] = np.mean(list(clustering_coeffs.values()))
                else:
                    features_dict['avg_clustering'] = clustering_coeffs
                
            features_list.append(np.array(list(features_dict.values())))
            
        return np.array(features_list)

class GraphNormalizer(Transformer):
    """Normalize graph adjacency matrices."""
    
    def __init__(self, normalization: str = 'symmetric'):
        """Initialize normalizer.
        
        Args:
            normalization: Normalization type ('symmetric' or 'random_walk')
        """
        valid_norms = {'symmetric', 'random_walk'}
        if normalization not in valid_norms:
            raise ValueError(f"normalization must be one of {valid_norms}")
            
        self.normalization = normalization
        
    def fit(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> 'GraphNormalizer':
        """Fit normalizer (no-op)."""
        return self
        
    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize adjacency matrices.
        
        Args:
            X: List of adjacency matrices
            
        Returns:
            List of normalized adjacency matrices
            
        Raises:
            ValidationError: If input matrices are invalid
        """
        if not X:
            raise ValidationError("Input list cannot be empty")
            
        normalized = []
        
        for adj in X:
            # Validate input
            if not isinstance(adj, np.ndarray):
                raise ValidationError(f"Expected numpy array, got {type(adj)}")
                
            if adj.shape[0] != adj.shape[1]:
                raise ValidationError("Adjacency matrix must be square")
                
            # Add self-loops to avoid isolated nodes
            adj = adj + np.eye(adj.shape[0])
            
            # Calculate degree matrix with epsilon to avoid division by zero
            D = np.diag(np.sum(adj, axis=1))
            D_data = np.maximum(D.diagonal(), EPSILON)
            
            if self.normalization == 'symmetric':
                # A' = D^(-1/2) A D^(-1/2)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(D_data))
                normalized.append(D_inv_sqrt @ adj @ D_inv_sqrt)
            else:  # random_walk
                # A' = D^(-1) A
                D_inv = np.diag(1.0 / D_data)
                normalized.append(D_inv @ adj)
                
        return normalized 