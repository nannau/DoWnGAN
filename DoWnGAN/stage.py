# Begin - load the data and initiate training
# Defines the hyperparameter and constants configurationsimport gc
from DoWnGAN.models.generator import Generator
from DoWnGAN.models.critic import Critic
from DoWnGAN.dataloader import NetCDFSR
import mlflow_utils as mlf 
import hyperparams as hp

from mlflow.tracking import MlflowClient
import torch

from helpers.gen_experiment_datasets import generate_train_test_coarse_fine

assert torch.cuda.is_available(), "CUDA not available"

# Load dataset
coarse_train, fine_train, coarse_test, fine_test = generate_train_test_coarse_fine()

# Covnert to tensors
coarse_train = torch.from_numpy(coarse_train.to_array().to_numpy()).transpose(0, 1).float()
fine_train = torch.from_numpy(fine_train.to_array().to_numpy()).transpose(0, 1).float()
coarse_test = torch.from_numpy(coarse_test.to_array().to_numpy()).transpose(0, 1).float()
fine_test = torch.from_numpy(fine_test.to_array().to_numpy()).transpose(0, 1).float()

# Get shapes for networks
fine_dim_n = fine_train.shape[-1]
n_predictands = fine_train.shape[1]
coarse_dim_n = coarse_train.shape[-1]
n_covariates = coarse_train.shape[1]

critic = Critic(coarse_dim_n, fine_dim_n, n_predictands).to(hp.device)
generator = Generator(coarse_dim_n, fine_dim_n, n_predictands).to(hp.device)

# Define optimizers
G_optimizer = torch.optim.Adam(generator.parameters(), hp.lr, betas=(0.9, 0.99))
C_optimizer = torch.optim.Adam(critic.parameters(), hp.lr, betas=(0.9, 0.99))

# Set up the run
# Define the mlflow experiment drectories
mlclient = MlflowClient(tracking_uri=hp.experiment_path)
exp_id = mlf.define_experiment(mlclient)
tag = mlf.write_tags()

# Definte the dataset objects
dataset = NetCDFSR(coarse_train, fine_train, device=hp.device)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=hp.batch_size, shuffle=True
)

testdataset = NetCDFSR(coarse_test, fine_test, device = hp.device)
testdataloader = torch.utils.data.DataLoader(
    dataset=testdataset, batch_size=hp.batch_size, shuffle=True
)

    
