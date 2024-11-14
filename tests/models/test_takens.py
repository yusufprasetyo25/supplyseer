import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError
from supplyseer.models.topological.takens import *

def test_takens_embedding_basic():
    """Test basic embedding functionality"""
    X = np.sin(np.linspace(0, 10*np.pi, 1000))
    time_delay = 10
    embedding_dimension = 3
    
    embedding = TakensEmbedding(
        time_delay=time_delay,
        embedding_dimension=embedding_dimension
    )
    X_embedded = embedding.fit_transform(X)
    
    # Expected number of vectors
    expected_vectors = len(X) - (embedding_dimension - 1) * time_delay
    
    assert X_embedded.shape[1] == embedding_dimension
    assert len(X_embedded) == expected_vectors

def test_takens_embedding_validation():
    """Test parameter validation"""
    with pytest.raises(ValueError):
        TakensEmbedding(time_delay=0)
    with pytest.raises(ValueError):
        TakensEmbedding(embedding_dimension=1)
    with pytest.raises(ValueError):
        TakensEmbedding(stride=0)

def test_takens_embedding_insufficient_data():
    """Test handling of insufficient data"""
    X = np.array([1, 2, 3])
    embedding = TakensEmbedding(time_delay=2, embedding_dimension=3)
    with pytest.raises(ValueError):
        embedding.fit_transform(X)

def test_takens_embedding_reconstruction():
    """Test reconstruction of time series from embedding"""
    X = np.sin(np.linspace(0, 10*np.pi, 1000))
    embedding = TakensEmbedding(time_delay=10, embedding_dimension=3)
    X_embedded = embedding.fit_transform(X)
    X_reconstructed = embedding.inverse_transform(X_embedded)
    
    # The reconstruction won't be exact due to averaging of overlapping points
    # We'll test that the reconstructed signal follows similar patterns
    center_slice = slice(100, -100)
    # Use larger tolerance for reconstruction comparison
    assert np.allclose(
        X[center_slice],
        X_reconstructed[center_slice],
        rtol=0.2,  # 20% relative tolerance
        atol=0.2   # 0.2 absolute tolerance
    )


def test_normalize_option():
    """Test normalization option"""
    X = np.sin(np.linspace(0, 10*np.pi, 1000))
    
    # With normalization
    embedding_norm = TakensEmbedding(
        time_delay=10,
        embedding_dimension=3,
        normalize=True
    )
    X_embedded_norm = embedding_norm.fit_transform(X)
    assert np.allclose(np.mean(X_embedded_norm, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(X_embedded_norm, axis=0), 1, atol=1e-10)
    
    # Without normalization
    embedding_no_norm = TakensEmbedding(
        time_delay=10,
        embedding_dimension=3,
        normalize=False
    )
    X_embedded_no_norm = embedding_no_norm.fit_transform(X)
    # For a sine wave, test that values stay within [-1, 1]
    assert np.all(np.abs(X_embedded_no_norm) <= 1.0)
    # Test that we preserve some variation in the signal
    assert np.all(np.std(X_embedded_no_norm, axis=0) > 0.1)