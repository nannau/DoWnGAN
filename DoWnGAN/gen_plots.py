import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mlflow


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
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value)  # (nxmx4)

    return value


def plot_to_tensorboard(fig, step):
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
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X).
    # Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.swapaxes(
        np.swapaxes(img, 0, 2), 1, 2
    )  # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    # writer.add_image('Results', img, step)
    plt.savefig("artifacts/image.png")
    plt.close(fig)


def generate_plots(fixed, path):
    # state = torch.load(loss_path)
    # losses_running = state["losses_running"]
    # loss_dict = state["losses_instance"]

    # writer.add_scalars('Losses', loss_dict, loss_dict["iters"])

    # fake = fixed["fake"]
    # fixed["real"] = fixed["real"]
    # coarse = fixed["coarse"]

    #     plt.style.use(['science','no-latex'])
    #     plt.rcParams.update({"figure.figsize":  (5,10),
    #                         'font.family': 'Times New Roman',
    #                         'font.size': 25,
    #                         'lines.linewidth': 2.5})

    fig, ax = plt.subplots(3, 5, figsize=(15, 15), subplot_kw=dict(box_aspect=1))

    cax = ax[0, 0].imshow(fixed["fake"][0, 0, ...], cmap="viridis", origin="lower")
    ax[0, 0].set_title("Generated")
    cax2 = ax[0, 1].imshow(fixed["real"][0, 0, ...], cmap="viridis", origin="lower")
    ax[0, 1].set_title("Ground Truth")
    cax3 = ax[0, 2].imshow(fixed["coarse"][0, 0, ...], cmap="viridis", origin="lower")
    ax[0, 2].set_title("Coarse")
    cax4 = ax[0, 3].imshow(fixed["low_real"][0, 0, ...], cmap="viridis", origin="lower")
    ax[0, 3].set_title("Low Pass Real")
    cax4 = ax[0, 4].imshow(fixed["low_fake"][0, 0, ...], cmap="viridis", origin="lower")
    ax[0, 4].set_title("Low Pass Fake")
    plt.colorbar(cax, ax=ax[0, 0], fraction=0.022)

    cax = ax[1, 0].imshow(fixed["fake"][1, 0, ...], cmap="viridis", origin="lower")
    ax[1, 0].set_title("Generated")
    cax2 = ax[1, 1].imshow(fixed["real"][1, 0, ...], cmap="viridis", origin="lower")
    ax[1, 1].set_title("Ground Truth")
    cax3 = ax[1, 2].imshow(fixed["coarse"][1, 0, ...], cmap="viridis", origin="lower")
    ax[1, 2].set_title("Coarse")
    cax4 = ax[1, 3].imshow(fixed["low_real"][1, 0, ...], cmap="viridis", origin="lower")
    ax[1, 3].set_title("Low Pass Real")
    cax4 = ax[1, 4].imshow(fixed["low_fake"][1, 0, ...], cmap="viridis", origin="lower")
    ax[1, 4].set_title("Low Pass Fake")
    plt.colorbar(cax, ax=ax[1, 0], fraction=0.022)

    cax = ax[2, 0].imshow(fixed["fake"][2, 0, ...], cmap="viridis", origin="lower")
    ax[2, 0].set_title("Generated")
    cax2 = ax[2, 1].imshow(fixed["real"][2, 0, ...], cmap="viridis", origin="lower")
    ax[2, 1].set_title("Ground Truth")
    cax3 = ax[2, 2].imshow(fixed["coarse"][2, 0, ...], cmap="viridis", origin="lower")
    ax[2, 2].set_title("Coarse")
    cax4 = ax[2, 3].imshow(fixed["low_real"][2, 0, ...], cmap="viridis", origin="lower")
    ax[2, 3].set_title("Low Pass Real")
    cax4 = ax[2, 4].imshow(fixed["low_fake"][2, 0, ...], cmap="viridis", origin="lower")
    ax[2, 4].set_title("Low Pass Fake")
    plt.colorbar(cax, ax=ax[2, 0], fraction=0.022)

    plt.tight_layout()
    plt.savefig("train_snap.png")
    plt.close(fig)
    mlflow.log_artifact("train_snap.png")
