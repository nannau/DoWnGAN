from DoWnGAN.losses import (
    content_loss,
    eof_loss,
    divergence_loss,
    vorticity_loss,
)
import pytest
import torch
import numpy as np

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
