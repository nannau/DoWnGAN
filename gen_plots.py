import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)
    
    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value) # (nxmx4)

    return value


def plot_to_tensorboard(writer, fig, step):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image('Results', img, step)
    plt.close(fig)


def write_to_board(writer, fixed, loss_path):
    state = torch.load(loss_path)
    losses_running = state["losses_running"]
    loss_dict = state["losses_instance"]

    writer.add_scalars('Losses', loss_dict, loss_dict["iters"])

    fake = fixed["fake"]
    real = fixed["real"]
    coarse = fixed["coarse"]

    plt.style.use(['science','no-latex'])
    plt.rcParams.update({"figure.figsize":  (5,10),
                        'font.family': 'Times New Roman',
                        'font.size': 25,
                        'lines.linewidth': 2.5})

    fig, ax = plt.subplots(2, 2, figsize=(15, 15), subplot_kw=dict(box_aspect=1))

    cax = ax[0, 0].imshow(fake[0, 0, ...], cmap="viridis")
    ax[0, 0].set_title("Generated")

    cax2 = ax[0, 1].imshow(real[0, 0, ...], cmap="viridis")
    ax[0, 1].set_title("Ground Truth")

    cax3 = ax[1, 0].imshow(coarse[0, 0, ...], cmap="viridis")
    ax[1, 0].set_title("Coarse")

    ax[1, 1].plot(losses_running["iters"], losses_running["pca"], label='pca')
    ax[1, 1].plot(losses_running["iters"], losses_running["G"], label='G')
    ax[1, 1].plot(losses_running["iters"], losses_running["D"], label='D')
    ax[1, 1].plot(losses_running["iters"], losses_running["Content"], label='content')
    ax[1, 1].set_title("Losses")
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_title("Loss")
    ax[1, 1].legend()

    plt.colorbar(cax, ax=ax[0, 0], fraction=0.022)
    plt.colorbar(cax2, ax=ax[0, 1], fraction=0.022)
    plt.colorbar(cax3, ax=ax[1, 0], fraction=0.022)

    plt.tight_layout()
    plot_to_tensorboard(writer, fig, loss_dict["iters"])
