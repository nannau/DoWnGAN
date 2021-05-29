import torch
import torch.nn as nn


@torch.jit.script
def content_loss(hr: torch.Tensor, fake: torch.Tensor, device: torch.device) -> float:
    """Calculates the L1 loss (pixel wise error) between both
    samples. Note that this is done on the high resolution (or super resolved fields)
    Args:
        hr (Tensor): Tensor containing batch of ground truth data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        content_loss (float): Single value corresponding to L1.
    """
    criterion_pixelwise = nn.L1Loss().to(device)
    content_loss = criterion_pixelwise(hr, fake)
    return content_loss


@torch.jit.script
def eof_loss(X: torch.Tensor, hr: torch.Tensor, fake: torch.Tensor, device: torch.device) -> float:
    """Calculates the L1 loss (EOF wise error) between the projections onto
    a subset of EOFs. 
    Args:
        X (Tensor): Tensor containing principle components of the input data
        hr (Tensor): Tensory containing batch of fake data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        closs (float): Loss between magnitudes of projection of ground truth
            compared with projection of fake
    """
    # Load PCA LHS
    # Reshape into batch size, channel size, n_pixels
    real = torch.reshape(
        hr, 
        (hr.size(0), hr.size(1), hr.size(2)*hr.size(3))
    ).unsqueeze(2).unsqueeze(1)

    # Reshape into batch size, channel size, n_pixels
    fake = torch.reshape(
        fake, 
        (fake.size(0), fake.size(1), fake.size(2)*fake.size(3))
    ).unsqueeze(2).unsqueeze(1)

    # Project onto the leading EOFs. Note that this is done for each member in the batch
    projected_real = torch.matmul(real, X.unsqueeze(-1))
    projected_fake = torch.matmul(fake, X.unsqueeze(-1))
    
    # Normalize by std
    std_real = torch.std(projected_real)
    std_fake = torch.std(projected_fake)
    
    projected_real = projected_real/std_real
    projected_fake = projected_fake/std_fake

    coefficient_loss = nn.MSELoss().to(device)
    closs = coefficient_loss(projected_fake, projected_real).item()
    return closs


@torch.jit.script
def divergence_loss(hr, fake, device):
    """Calculates the L1 loss (pixel wise error) between divergence of both samples
    Note that this is done on the high resolution (or super resolved fields). Channel 0 
    on colour axis is u10, and channel 1 on colour axis is v10.
    Args:
        hr (Tensor): Tensor containing batch of ground truth data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        divergence_loss (float): Single value corresponding to L1 loss between respective
            divergences
    """
    # 0 on color axis (1) is u10
    # 1 on color axis (1) is v10
    # Calculate difference across latitude and longitude
    # note that this is not divided by the change in latitude/longitude
    # due to regular grids
    dudy_real = hr[:, 0, 1:, 1:] - hr[:, 0, :-1, 1:]
    dvdx_real = hr[:, 1, 1:, 1:] - hr[:, 1, 1:, :-1]
    # Divergence
    div_real = dudy_real + dvdx_real

    dudy_fake = fake[:, 0, 1:, 1:] - fake[:, 0, :-1, 1:]
    dvdx_fake = fake[:, 1, 1:, 1:] - fake[:, 1, 1:, :-1]
    # Divergence
    div_fake = dudy_fake + dvdx_fake

    std_norm_real = torch.std(div_real)
    std_norm_fake = torch.std(div_fake)

    div_real = div_real/std_norm_real
    div_fake = div_fake/std_norm_fake

    divergence_loss = nn.MSELoss().to(device)

    return divergence_loss(div_real, div_fake).item()

@torch.jit.script
def vorticity_loss(hr, fake, device):
    """Calculates the L1 loss (pixel wise error) between vorticity of both samples
    Note that this is done on the high resolution (or super resolved fields). Channel 0 
    on colour axis is u10, and channel 1 on colour axis is v10.
    Args:
        hr (Tensor): Tensor containing batch of ground truth data
        fake (Tensor): Tensory containing batch of fake data
        device: device to be run on
    Returns:
        vort_loss (float): Single value corresponding to L1 
            loss between respective vorticities.
    """
    # 0 on color axis (1) is u10
    # 1 on color axis (1) is v10
    # Calculate difference across latitude and longitude
    # note that this is not divided by the change in latitude/longitude
    # due to regular grids
    dudy_real = hr[:, 0, 1:, 1:] - hr[:, 0, :-1, 1:]
    dvdx_real = hr[:, 1, 1:, 1:] - hr[:, 1, 1:, :-1]
    # Vorticity
    vort_real = dvdx_real - dudy_real

    dudy_fake = fake[:, 0, 1:, 1:] - fake[:, 0, :-1, 1:]
    dvdx_fake = fake[:, 1, 1:, 1:] - fake[:, 1, 1:, :-1]
    # Vorticity
    vort_fake = dvdx_fake - dudy_fake

    std_norm_real = torch.std(vort_real)
    std_norm_fake = torch.std(vort_fake)

    vort_real = vort_real/std_norm_real
    vort_fake = vort_fake/std_norm_fake

    vort_loss = nn.MSELoss().to(device)

    return vort_loss(vort_real, vort_fake).item()