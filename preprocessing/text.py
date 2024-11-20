"""Text preprocessing utilities."""

import numpy as np
from typing import List, Dict, Optional, Union
from collections import Counter
import re
from core import (
    Transformer,
    check_array,
    get_logger,
    ValidationError
)

class TextPreprocessor(Transformer):
    """Basic text preprocessing utilities."""
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = True,
                 stem: bool = False,
                 lemmatize: bool = False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.lemmatize = lemmatize
        self._stopwords = set()  # Will be loaded on first use
        
    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> 'TextPreprocessor':
        """Fit preprocessor (load resources if needed)."""
        if self.remove_stopwords:
            try:
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words('english'))
            except:
                import nltk
                nltk.download('stopwords')
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words('english'))
        return self
        
    def transform(self, X: List[str]) -> List[str]:
        """Transform text data."""
        processed = []
        for text in X:
            if self.lowercase:
                text = text.lower()
            
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s]', '', text)
                
            if self.remove_numbers:
                text = re.sub(r'\d+', '', text)
                
            tokens = text.split()
            
            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in self._stopwords]
                
            if self.stem:
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(t) for t in tokens]
                
            if self.lemmatize:
                from nltk.stem import WordNetLemmatizer
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
                
            processed.append(' '.join(tokens))
            
        return processed

class TfidfVectorizer(Transformer):
    """Convert text collection to TF-IDF features."""
    
    def __init__(self,
                 max_features: Optional[int] = None,
                 min_df: Union[int, float] = 1,
                 max_df: Union[int, float] = 1.0,
                 ngram_range: tuple = (1, 1)):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vocabulary_ = None
        self.idf_ = None
        
    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> 'TfidfVectorizer':
        """Learn vocabulary and IDF values."""
        # Build vocabulary
        word_counts = Counter()
        doc_counts = Counter()
        n_docs = len(X)
        
        for doc in X:
            words = set(doc.split())  # Unique words in document
            doc_counts.update(words)
            word_counts.update(doc.split())
            
        # Filter by document frequency
        min_docs = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_docs = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        
        valid_words = {
            word for word, count in doc_counts.items()
            if min_docs <= count <= max_docs
        }
        
        if self.max_features:
            valid_words = {
                word for word, _ in sorted(
                    ((w, c) for w, c in word_counts.items() if w in valid_words),
                    key=lambda x: (-x[1], x[0])
                )[:self.max_features]
            }
            
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(valid_words))}
        
        # Calculate IDF
        self.idf_ = np.zeros(len(self.vocabulary_))
        for word, idx in self.vocabulary_.items():
            self.idf_[idx] = np.log(n_docs / (doc_counts[word] + 1)) + 1
            
        return self
        
    def transform(self, X: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF matrix."""
        if self.vocabulary_ is None or self.idf_ is None:
            raise ValueError("Vocabulary not fitted. Call fit() first.")
            
        n_docs = len(X)
        n_features = len(self.vocabulary_)
        result = np.zeros((n_docs, n_features))        
        for doc_idx, doc in enumerate(X):
            word_counts = Counter(doc.split())
            doc_length = sum(word_counts.values())
            
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    word_idx = self.vocabulary_[word]
                    tf = count / doc_length
                    result[doc_idx, word_idx] = tf * self.idf_[word_idx]
                    
        return result

class Word2VecEncoder(Transformer):
    """Encode text using Word2Vec embeddings."""
    
    def __init__(self,
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 5,
                 workers: int = 4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        
    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> 'Word2VecEncoder':
        """Train Word2Vec model."""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("Please install gensim to use Word2Vec encoding")
            
        # Tokenize documents
        sentences = [doc.split() for doc in X]
        
        # Train model
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        
        return self
        
    def transform(self, X: List[str]) -> np.ndarray:
        """Transform documents to averaged word vectors."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        result = np.zeros((len(X), self.vector_size))
        
        for idx, doc in enumerate(X):
            vectors = []
            for word in doc.split():
                if word in self.model.wv:
                    vectors.append(self.model.wv[word])
            if vectors:
                result[idx] = np.mean(vectors, axis=0)
                
        return result 