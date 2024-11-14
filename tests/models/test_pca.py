import numpy as np

from supplyseer.models.statistical.pca import PCA


def test_pca():

    x = np.linspace(1, 1000, 1000).reshape(-1, 10)

    xmax_expected, xmin_expected = 1000, 1
    
    assert x.max() == xmax_expected
    assert x.min() == xmin_expected

    pca = PCA(x)
    pca_output = pca.fit_pca(k=3)

    assert type(pca_output) == dict
    assert pca_output["n_embeddings"] == 3
    assert pca_output["pca_data"].shape == (100, 3)
    assert pca_output["explained_variance"].shape == (10,)

    assert np.round(pca_output["pca_data"].max()) == 1565.0
    assert np.round(pca_output["pca_data"].min()) == -1565.0
    assert np.round(pca_output["explained_variance"].sum()) == 1.0, "Sum of explained variance is not 1.0"