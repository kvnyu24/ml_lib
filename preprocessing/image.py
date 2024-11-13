"""Image preprocessing utilities."""

import numpy as np
from typing import Optional, Tuple, Union
from core import BaseTransformer
import cv2

class ImagePreprocessor(BaseTransformer):
    """Basic image preprocessing utilities."""
    
    def __init__(self,
                 target_size: Optional[Tuple[int, int]] = None,
                 normalize: bool = True,
                 grayscale: bool = False,
                 augment: bool = False):
        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale
        self.augment = augment
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ImagePreprocessor':
        """Fit preprocessor (no-op)."""
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform images."""
        processed = []
        
        for img in X:
            if self.grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.expand_dims(img, axis=-1)
                
            if self.target_size:
                img = cv2.resize(img, self.target_size)
                
            if self.normalize:
                img = img.astype(np.float32) / 255.0
                
            if self.augment:
                img = self._augment_image(img)
                
            processed.append(img)
            
        return np.array(processed)
        
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            
        # Random rotation
        angle = np.random.uniform(-15, 15)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        return img

class ImageFeatureExtractor(BaseTransformer):
    """Extract features from images using pre-trained models."""
    
    def __init__(self,
                 model_name: str = 'resnet50',
                 pooling: str = 'avg',
                 include_top: bool = False):
        self.model_name = model_name
        self.pooling = pooling
        self.include_top = include_top
        self.model = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ImageFeatureExtractor':
        """Load pre-trained model."""
        try:
            from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
        except ImportError:
            raise ImportError("Please install tensorflow to use pre-trained models")
            
        models = {
            'resnet50': ResNet50,
            'vgg16': VGG16,
            'mobilenetv2': MobileNetV2
        }
        
        if self.model_name not in models:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        self.model = models[self.model_name](
            include_top=self.include_top,
            weights='imagenet',
            pooling=self.pooling
        )
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract features using pre-trained model."""
        # Ensure input format matches model requirements
        if self.model_name in ['resnet50', 'vgg16']:
            from tensorflow.keras.applications.resnet50 import preprocess_input
        else:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            
        X = preprocess_input(X)
        features = self.model.predict(X)
        
        return features 