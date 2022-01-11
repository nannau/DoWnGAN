# Defines the hyperparameter and constants configurationsimport gc
from DoWnGAN.losses import (
    content_loss,
    content_MSELoss,
    SSIM_Loss,
    wass_loss
)
import torch
import torch.nn as nn

import os

device = torch.device("cuda:0")

# Hyper params
gp_lambda = 10
critic_iterations = 5
batch_size = 64
gamma = 0.01
content_lambda = 0.0 #5
ncomp = 75
lr = 0.00025

# Run configuration parameters
epochs = 1000
print_every = 250
save_every = 250
use_cuda = True
device = device

# Frequency separation parameters
freq_sep = False
filter_size = 5
padding = filter_size // 2
low = nn.AvgPool2d(filter_size, stride=1, padding=0)
rf = nn.ReplicationPad2d(padding)

# Whether to load preprocessed data
already_preprocessed = True

experiment_path = os.getenv("EXPERIMENT_PATH")

metrics_to_calculate = {
    "MAE": content_loss,
    "MSE": content_MSELoss,
    "MSSSIM": SSIM_Loss,
    "Wass": wass_loss
}
