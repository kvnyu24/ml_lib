"""Latent Dirichlet Allocation implementation."""

import numpy as np
from typing import Optional, List, Tuple, Dict
from core.base import Estimator
from core.validation import check_array, check_X_y
from scipy.special import digamma, gammaln
import logging

logger = logging.getLogger(__name__)

class LatentDirichletAllocation(Estimator):
    """Latent Dirichlet Allocation using variational inference.
    
    Implements LDA using mean-field variational inference for topic modeling.
    """
    
    def __init__(self,
                 n_components: int = 10,
                 doc_topic_prior: float = None,
                 topic_word_prior: float = None,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 random_state: Optional[int] = None):
        """Initialize LDA.
        
        Args:
            n_components: Number of topics
            doc_topic_prior: Dirichlet prior on document-topic distribution
            topic_word_prior: Dirichlet prior on topic-word distribution
            max_iter: Maximum number of iterations
            tol: Tolerance for stopping criterion
            random_state: Random state for reproducibility
        """
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior or 1/n_components
        self.topic_word_prior = topic_word_prior or 1/n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Attributes set during fit
        self.components_ = None  # topic-word distribution
        self.doc_topic_dist_ = None  # document-topic distribution
        self.n_iter_ = 0
        
    def _init_latent_vars(self, n_docs: int, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize latent variables."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Initialize document-topic distribution
        doc_topic_dist = np.random.gamma(100., 0.01, (n_docs, self.n_components))
        doc_topic_dist /= doc_topic_dist.sum(axis=1)[:, np.newaxis]
        
        # Initialize topic-word distribution
        topic_word_dist = np.random.gamma(100., 0.01, (self.n_components, vocab_size))
        topic_word_dist /= topic_word_dist.sum(axis=1)[:, np.newaxis]
        
        return doc_topic_dist, topic_word_dist
        
    def _e_step(self, X: np.ndarray, doc_topic_dist: np.ndarray,
                topic_word_dist: np.ndarray) -> Tuple[np.ndarray, float]:
        """E-step: update document-topic distribution."""
        n_docs, vocab_size = X.shape
        
        # Compute document-topic sufficient statistics
        doc_topic_dist_new = np.zeros_like(doc_topic_dist)
        bound = 0.
        
        for d in range(n_docs):
            # Compute variational expectation
            ids = np.nonzero(X[d, :])[0]
            counts = X[d, ids]
            
            if len(ids) == 0:
                continue
                
            # Update document-topic distribution
            exp_topic_word = topic_word_dist[:, ids]
            exp_doc_topic = np.exp(digamma(doc_topic_dist[d]) - 
                                 digamma(doc_topic_dist[d].sum()))
                                 
            norm = exp_doc_topic.dot(exp_topic_word) + 1e-100
            
            doc_topic_ss = exp_doc_topic[:, np.newaxis] * exp_topic_word / norm[np.newaxis, :]
            doc_topic_dist_new[d] = self.doc_topic_prior + np.sum(counts * doc_topic_ss, axis=1)
            
            # Update bound
            bound += np.sum(counts * np.log(norm))
            
        return doc_topic_dist_new, bound
        
    def _m_step(self, X: np.ndarray, doc_topic_dist: np.ndarray) -> np.ndarray:
        """M-step: update topic-word distribution."""
        n_docs, vocab_size = X.shape
        topic_word_dist = np.zeros((self.n_components, vocab_size))
        
        # Update topic-word distribution
        for d in range(n_docs):
            ids = np.nonzero(X[d, :])[0]
            counts = X[d, ids]
            
            if len(ids) > 0:
                exp_doc_topic = np.exp(digamma(doc_topic_dist[d]) - 
                                     digamma(doc_topic_dist[d].sum()))
                topic_word_dist[:, ids] += np.outer(exp_doc_topic, counts)
                
        # Normalize
        topic_word_dist += self.topic_word_prior
        topic_word_dist /= topic_word_dist.sum(axis=1)[:, np.newaxis]
        
        return topic_word_dist
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LatentDirichletAllocation':
        """Fit LDA model.
        
        Args:
            X: Document-term matrix (n_samples, n_features)
            y: Ignored
            
        Returns:
            self: Fitted estimator
        """
        X = check_array(X, accept_sparse=True)
        n_docs, vocab_size = X.shape
        
        # Initialize
        doc_topic_dist, topic_word_dist = self._init_latent_vars(n_docs, vocab_size)
        
        # EM algorithm
        bound = -np.inf
        for n_iter in range(self.max_iter):
            # E-step
            doc_topic_dist_new, bound_new = self._e_step(X, doc_topic_dist, topic_word_dist)
            
            # M-step
            topic_word_dist = self._m_step(X, doc_topic_dist_new)
            
            # Check convergence
            bound_improvement = (bound_new - bound) / abs(bound)
            
            if bound_improvement < self.tol:
                break
                
            bound = bound_new
            doc_topic_dist = doc_topic_dist_new
            self.n_iter_ += 1
            
            if self.n_iter_ % 10 == 0:
                logger.info(f'Iteration {self.n_iter_}/{self.max_iter}')
                logger.info(f'Bound improvement: {bound_improvement:.6f}')
        
        self.components_ = topic_word_dist
        self.doc_topic_dist_ = doc_topic_dist
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new documents to topic distributions.
        
        Args:
            X: Document-term matrix (n_samples, n_features)
            
        Returns:
            Document-topic distribution for X
        """
        X = check_array(X, accept_sparse=True)
        doc_topic_dist, _ = self._init_latent_vars(X.shape[0], X.shape[1])
        
        # Single E-step to get document-topic distribution
        doc_topic_dist, _ = self._e_step(X, doc_topic_dist, self.components_)
        
        return doc_topic_dist