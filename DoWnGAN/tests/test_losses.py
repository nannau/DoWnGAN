from DoWnGAN.losses import (
    content_loss,
    eof_loss,
    divergence_loss,
    vorticity_loss,
)
import pytest
import torch
import numpy as np
import xarray as xr

from DoWnGAN.dataloader import xr_standardize_field
from sklearn.decomposition import PCA
from pkg_resources import resource_filename
from sklearn.metrics import mean_squared_error


# Shape is:
# batch_size, channels, x, y

N = 5
hr = torch.ones((64, 2, 10, 12))
fake = N * torch.ones((64, 2, 10, 12))

hrg = torch.reshape(torch.arange(64 * 2 * 10 * 12), hr.size()).float()
fakeg = torch.reshape(torch.arange(64 * 2 * 10 * 12), fake.size()).float()

xx, yy = torch.meshgrid(torch.arange(-5, 5), torch.arange(-6, 6))
zz_hr = torch.exp(-(xx ** 2 + yy ** 2))
zz_fake = torch.exp(-(xx ** 4 + yy ** 4))

# Add gaussian to channels
for i in range(hr.size(0)):
    hrg[i, 0, :, :] = zz_hr
    hrg[i, 1, :, :] = zz_hr
    fakeg[i, 0, :, :] = zz_fake
    fakeg[i, 1, :, :] = zz_fake

device = torch.device("cpu")


@pytest.fixture
def example_data():
    return {"content": [hr, fake, device], "grads": [hrg, fakeg, device]}

@pytest.fixture
def actual_data():
    dataroot = resource_filename("DoWnGAN", "tests/coarse_test.nc")
    real = xr.open_dataset(dataroot)
    X = real["u10"][:100, ...]
    X = np.array(xr_standardize_field(X))
    return {"data": X}


def test_content_loss(example_data):
    # Load data
    hr, fake, device = example_data["content"]

    # Types
    assert isinstance(hr, torch.Tensor)
    assert isinstance(fake, torch.Tensor)
    assert isinstance(device, torch.device)

    # Data

    # Shapes
    assert hr.size() == fake.size()

    # Value
    closs = content_loss(hr, fake, device)
    closs_truth = N - 1
    assert isinstance(float(closs), float)


def test_divergence_loss(example_data):
    # Load data
    hr, fake, device = example_data["grads"]

    # Types
    assert isinstance(hr, torch.Tensor)
    assert isinstance(fake, torch.Tensor)
    assert isinstance(device, torch.device)

    # Data

    # Shapes
    assert hr.size() == fake.size()

    # Value
    dloss = divergence_loss(hr, fake, device)
    dloss_truth = 0.0018

    assert isinstance(dloss, float)
    assert np.isclose(dloss, dloss_truth, atol=0.0001)


def test_vorticity_loss(example_data):
    # Load data
    hr, fake, device = example_data["grads"]

    # Types
    assert isinstance(hr, torch.Tensor)
    assert isinstance(fake, torch.Tensor)
    assert isinstance(device, torch.device)

    # Data

    # Shapes
    assert hr.size() == fake.size()

    # Value
    vloss = vorticity_loss(hr, fake, device)
    vloss_truth = 0.00144

    assert isinstance(vloss, float)
    assert np.isclose(vloss, vloss_truth, atol=0.0001)


def test_eof_loss(actual_data):

    n_comps = 20

    # Load data
    X = actual_data["data"]

    # Get EOFs
    pca = PCA(n_comps)
    pca.fit(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))

    # Normalize by magnitudes of eigenvalues
    Z = (
        pca.components_.reshape(n_comps, X.shape[1] * X.shape[2])
        / pca.explained_variance_[:, np.newaxis]
    )

    # Convert to tensors
    X = torch.from_numpy(X)
    Z = torch.from_numpy(Z)

    # Reshape to project
    X = torch.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
    Z = Z.unsqueeze(-1)

    # Corrupt a signal to test against magnitudes
    X_c = torch.clone(X)
    X_c[0, ...] = X[0, ...] + np.random.normal(scale=5, size=(X.shape[1]))

    magnitude = torch.matmul(X, Z)
    magnitude_c = torch.matmul(X_c, Z)

    assert X_c.size() == X.size()
    # Test corrupted signal is non-zero difference over all
    # EOFs
    assert not np.isclose(
        mean_squared_error(magnitude[:, :, 0], magnitude_c[:, :, 0]), 0.0, atol=1e-6
    )
    # Tests that non-corrupted signal is zero
    assert np.isclose(mean_squared_error(magnitude[:, :, 0], magnitude[:, :, 0]), 0.0)
