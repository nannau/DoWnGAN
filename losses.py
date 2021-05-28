import torch
import torch.nn as nn


def content_loss(hr, generated_data, device):
    # Question is content loss compared to 
    # LR data or HR data? I've seen both.

    # Get generated data
    # avg_pool = nn.AvgPool2d(4, stride=4)
    # gen_lr = avg_pool(generated_data)

    criterion_pixelwise = nn.L1Loss().to(device)

    # content_loss = criterion_pixelwise(cr, gen_lr)
    content_loss = criterion_pixelwise(hr, generated_data)

    return content_loss

def eof_loss(X, hr, fake, device):

    # Load PCA LHS
    real = torch.reshape(
        hr, 
        (hr.size(0), hr.size(1), hr.size(2)*hr.size(3))
    ).unsqueeze(2).unsqueeze(1)

    fake = torch.reshape(
        fake, 
        (fake.size(0), fake.size(1), fake.size(2)*fake.size(3))
    ).unsqueeze(2).unsqueeze(1)

    projected_real = torch.matmul(real, X.unsqueeze(-1))
    projected_fake = torch.matmul(fake, X.unsqueeze(-1))
    
    std_real = torch.std(projected_real)
    std_fake = torch.std(projected_fake)
    
    projected_real = projected_real/std_real
    projected_fake = projected_fake/std_fake

    coefficient_loss = nn.MSELoss().to(device)
    closs = coefficient_loss(projected_fake, projected_real).item()
    return closs


def divergence_loss(hr, fake, device):
    # 0 on color axis (1) is u10
    # 1 on color axis (1) is v10
    dudy_real = hr[:, 0, 1:, 1:] - hr[:, 0, :-1, 1:]
    dvdx_real = hr[:, 1, 1:, 1:] - hr[:, 1, 1:, :-1]
    div_real = dudy_real + dvdx_real

    dudy_fake = fake[:, 0, 1:, 1:] - fake[:, 0, :-1, 1:]
    dvdx_fake = fake[:, 1, 1:, 1:] - fake[:, 1, 1:, :-1]
    div_fake = dudy_fake + dvdx_fake

    std_norm_real = torch.std(div_real)
    std_norm_fake = torch.std(div_fake)

    div_real = div_real/std_norm_real
    div_fake = div_fake/std_norm_fake

    divergence_loss = nn.MSELoss().to(device)

    return divergence_loss(div_real, div_fake).item()

def vorticity_loss(hr, fake, device):
    # 0 on color axis (1) is u10
    # 1 on color axis (1) is v10
    dudy_real = hr[:, 0, 1:, 1:] - hr[:, 0, :-1, 1:]
    dvdx_real = hr[:, 1, 1:, 1:] - hr[:, 1, 1:, :-1]
    vort_real = dvdx_real - dudy_real

    dudy_fake = fake[:, 0, 1:, 1:] - fake[:, 0, :-1, 1:]
    dvdx_fake = fake[:, 1, 1:, 1:] - fake[:, 1, 1:, :-1]
    vort_fake = dvdx_fake - dudy_fake

    std_norm_real = torch.std(vort_real)
    std_norm_fake = torch.std(vort_fake)

    vort_real = vort_real/std_norm_real
    vort_fake = vort_fake/std_norm_fake

    vorticity_loss = nn.MSELoss().to(device)

    return vorticity_loss(vort_real, vort_fake).item()