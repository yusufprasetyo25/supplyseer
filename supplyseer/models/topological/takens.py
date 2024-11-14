import numpy as np
from typing import Optional, Tuple, Union, List
from pydantic import BaseModel, Field
from sklearn.utils import column_or_1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors

class TakensEmbeddingConfig(BaseModel):
    """Configuration for Takens embedding parameters"""
    time_delay: int = Field(1, gt=0, description="Time delay between successive coordinates")
    embedding_dimension: int = Field(2, gt=1, description="Number of embedding dimensions")
    stride: int = Field(1, gt=0, description="Step size between consecutive vectors")
    normalize: bool = Field(True, description="Whether to normalize the embedded vectors")
    
    class Config:
        arbitrary_types_allowed = True

class TakensEmbedding(BaseEstimator, TransformerMixin):
    """
    Implements Takens' embedding theorem for time series reconstruction.
    
    This class transforms a univariate time series into its phase space representation
    using time-delay embedding, as per Takens' theorem.
    
    Parameters
    ----------
    time_delay : int, default=1
        Time delay between successive coordinates
    embedding_dimension : int, default=2
        Number of embedding dimensions
    stride : int, default=1
        Step size between consecutive vectors
    normalize : bool, default=True
        Whether to normalize the embedded vectors
        
    Attributes
    ----------
    config_ : TakensEmbeddingConfig
        Validated configuration parameters
    n_features_in_ : int
        Number of features seen during fit
    """
    
    def __init__(self, 
                 time_delay: int = 1,
                 embedding_dimension: int = 2,
                 stride: int = 1,
                 normalize: bool = True):
        self.config_ = TakensEmbeddingConfig(
            time_delay=time_delay,
            embedding_dimension=embedding_dimension,
            stride=stride,
            normalize=normalize
        )
        self.n_features_in_ = None
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate the parameters"""
        if self.config_.time_delay < 1:
            raise ValueError("time_delay must be at least 1")
        if self.config_.embedding_dimension < 2:
            raise ValueError("embedding_dimension must be at least 2")
        if self.config_.stride < 1:
            raise ValueError("stride must be at least 1")
            
    def _check_sufficient_length(self, X):
        """Check if time series is long enough for embedding"""
        min_length = (self.config_.embedding_dimension - 1) * self.config_.time_delay + 1
        if len(X) < min_length:
            raise ValueError(
                f"Time series length must be at least {min_length} for the current "
                f"embedding parameters. Got length {len(X)}."
            )

    def fit(self, X: np.ndarray, y=None) -> 'TakensEmbedding':
        """
        Fit the embedding parameters to the data.
        Currently just validates input shape.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series
        y : None
            Ignored, exists for scikit-learn compatibility
            
        Returns
        -------
        self : object
            Returns self
        """
        X = column_or_1d(X)
        self._check_sufficient_length(X)
        self.n_features_in_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to its embedded representation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series
            
        Returns
        -------
        X_embedded : ndarray of shape (n_vectors, embedding_dimension)
            Embedded time series
        """
        # Validate input
        X = column_or_1d(X)
        self._check_sufficient_length(X)
        
        # Calculate embedding parameters
        n_samples = len(X)
        n_vectors = (n_samples - (self.config_.embedding_dimension - 1) * 
                    self.config_.time_delay - 1) // self.config_.stride + 1
        
        # Initialize embedded matrix
        X_embedded = np.zeros((n_vectors, self.config_.embedding_dimension))
        
        # Create embedded vectors
        for i in range(self.config_.embedding_dimension):
            start_idx = i * self.config_.time_delay
            end_idx = start_idx + (n_vectors - 1) * self.config_.stride + 1
            X_embedded[:, i] = X[start_idx:end_idx:self.config_.stride]
        
        # Normalize if requested
        if self.config_.normalize:
            X_embedded = (X_embedded - np.mean(X_embedded, axis=0)) / np.std(X_embedded, axis=0)
            
        return X_embedded

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit and transform the time series.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series
        y : None
            Ignored, exists for scikit-learn compatibility
            
        Returns
        -------
        X_embedded : ndarray of shape (n_vectors, embedding_dimension)
            Embedded time series
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_embedded: np.ndarray) -> np.ndarray:
        """
        Reconstruct time series from its embedded representation.
        Uses averaging of overlapping coordinates.
        
        Parameters
        ----------
        X_embedded : ndarray of shape (n_vectors, embedding_dimension)
            Embedded time series
            
        Returns
        -------
        X_reconstructed : ndarray
            Reconstructed time series
        """
        if self.config_.normalize:
            X_embedded = X_embedded * np.std(X_embedded, axis=0) + np.mean(X_embedded, axis=0)
            
        n_vectors = X_embedded.shape[0]
        series_length = (n_vectors - 1) * self.config_.stride + \
                       (self.config_.embedding_dimension - 1) * self.config_.time_delay + 1
        
        X_reconstructed = np.zeros(series_length)
        counts = np.zeros(series_length)
        
        # Reconstruct by averaging overlapping coordinates
        for i in range(n_vectors):
            for j in range(self.config_.embedding_dimension):
                idx = i * self.config_.stride + j * self.config_.time_delay
                X_reconstructed[idx] += X_embedded[i, j]
                counts[idx] += 1
                
        # Average where there are multiple contributions
        X_reconstructed[counts > 0] /= counts[counts > 0]
        
        return X_reconstructed
    
    def estimate_optimal_parameters(self, X: np.ndarray, 
                                  max_time_delay: int = 20,
                                  max_embedding_dim: int = 10) -> Tuple[int, int]:
        """
        Estimate optimal time delay and embedding dimension.
        
        Uses mutual information for time delay and false nearest neighbors
        for embedding dimension.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series
        max_time_delay : int, default=20
            Maximum time delay to consider
        max_embedding_dim : int, default=10
            Maximum embedding dimension to consider
            
        Returns
        -------
        optimal_delay : int
            Estimated optimal time delay
        optimal_dim : int
            Estimated optimal embedding dimension
        """
        X = column_or_1d(X)
        
        # Estimate optimal time delay using mutual information
        mi_scores = []
        for delay in range(1, max_time_delay + 1):
            X_delayed = X[delay:]
            X_current = X[:-delay]
            # Discretize the data for MI calculation
            bins = min(int(np.sqrt(len(X))), 100)
            X_binned = np.searchsorted(np.linspace(X.min(), X.max(), bins), X_current)
            Xd_binned = np.searchsorted(np.linspace(X.min(), X.max(), bins), X_delayed)
            mi = mutual_info_score(X_binned, Xd_binned)
            mi_scores.append(mi)
        
        optimal_delay = np.argmin(mi_scores) + 1
        
        # Estimate optimal embedding dimension using false nearest neighbors
        fnn_ratios = []
        for dim in range(2, max_embedding_dim + 1):
            # Create embedded vectors for dimensions d and d+1
            embedding_d = TakensEmbedding(
                time_delay=optimal_delay,
                embedding_dimension=dim,
                stride=1,
                normalize=True
            ).fit_transform(X)
            
            embedding_d1 = TakensEmbedding(
                time_delay=optimal_delay,
                embedding_dimension=dim+1,
                stride=1,
                normalize=True
            ).fit_transform(X)
            
            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(embedding_d)
            distances, indices = nn.kneighbors(embedding_d)
            
            # Calculate FNN ratio
            Ra = np.std(X)
            fnn = 0
            for i in range(len(embedding_d)):
                neighbor_idx = indices[i, 1]
                d_distance = distances[i, 1]
                if d_distance == 0:
                    continue
                    
                d1_distance = np.abs(embedding_d1[i, -1] - embedding_d1[neighbor_idx, -1])
                if d1_distance / d_distance > 10 or d1_distance / Ra > 2:
                    fnn += 1
                    
            fnn_ratios.append(fnn / len(embedding_d))
        
        # Find first dimension where FNN ratio drops below threshold
        threshold = 0.1
        optimal_dim = np.where(np.array(fnn_ratios) < threshold)[0]
        optimal_dim = optimal_dim[0] + 2 if len(optimal_dim) > 0 else max_embedding_dim
        
        return optimal_delay, optimal_dim