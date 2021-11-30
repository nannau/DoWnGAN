# Plots matplotlib grids and saves to file
import torch
import matplotlib.pyplot as plt
import torchvision
import hyperparams as hp
import mlflow

def gen_grid_images(G, coarse, real, epoch, train_test):
    """
    Plots a grid of images and saves them to file
    Args:
        coarse (torch.Tensor): The coarse input.
        fake (torch.Tensor): The fake input.
        real (torch.Tensor): The real input.
    """
    torch.manual_seed(0)
    random = torch.randint(0, hp.batch_size, (20, ))
    fake = G(coarse[random, ...].to(hp.device))
    coarse = torchvision.utils.make_grid(
        coarse[random, ...],
        nrow=10,
        padding=5
    )[0, ...]

    fake = torchvision.utils.make_grid(
        fake,
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
    ax.imshow(coarse.cpu().detach(), origin="lower")

    # Generated fake
    subfigs[1].suptitle("Generated Fields")
    ax = subfigs[1].subplots(1, 1)
    ax.imshow(fake.cpu().detach(), origin="lower")

    # Ground Truth
    subfigs[2].suptitle("Ground Truth")
    ax = subfigs[2].subplots(1, 1)
    ax.imshow(real.cpu().detach(), origin="lower")

    if epoch % 10 == 0:
        plt.savefig(f"{mlflow.get_artifact_uri()}/{train_test}_{epoch}.png")
    else:
        plt.savefig(f"{train_test}.png")
        mlflow.log_artifact(f"{train_test}.png")
    plt.close(fig)