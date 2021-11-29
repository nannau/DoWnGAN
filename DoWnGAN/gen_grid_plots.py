# Plots matplotlib grids and saves to file
import torch
import matplotlib.pyplot as plt
import torchvision
import hyperparams as hp
import mlflow

def gen_grid_images(G, coarse, real, filename):
    """
    Plots a grid of images and saves them to file
    Args:
        coarse (torch.Tensor): The coarse input.
        fake (torch.Tensor): The fake input.
        real (torch.Tensor): The real input.
    """
    random = torch.randn(0, hp.batch_size, 20)
    coarse = torchvision.utils.make_grid(
        coarse[random, ...],
        nrow=10
    )[0, ...]

    fake = G(coarse)
    fake = torchvision.utils.make_grid(
        fake[random, ...],
        nrow=10
    )[0, ...]

    real = torchvision.utils.make_grid(
        real[random, ...],
        nrow=10
    )[0, ...]


    fig = plt.figure(figsize=(30, 10))
    fig.suptitle("Training Samples")

    # Plot the coarse and fake samples
    subfigs = fig.subfigures(nrows=3, ncols=1)
    
    # Coarse Samples
    subfigs[0].suptitle("Coarse ERAI")
    ax = subfigs[0].subplots(1, 1)
    ax.imshow(coarse.detach(), origin="lower")

    # Generated fake
    subfigs[1].suptitle("Generated Fields")
    ax = subfigs[1].subplots(1, 1)
    ax.imshow(fake.detach(), origin="lower")

    # Ground Truth
    subfigs[2].suptitle("Ground Truth")
    ax = subfigs[2].subplots(1, 1)
    ax.imshow(real.detach(), origin="lower")

    plt.tight_layout()
    plt.savefig("train_snap.png")
    plt.close(fig)
    mlflow.log_artifact(f"{filename}.png")