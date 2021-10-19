import gc
import pickle
import glob
import os

from mlflow.tracking import MlflowClient
import mlflow
import xarray as xr
import numpy as np
import pandas as pd
import torch
import dask
from pkg_resources import resource_filename
from dask.distributed import Client
from sklearn.metrics import mean_squared_error
from scipy.interpolate import NearestNDInterpolator

import matplotlib.pyplot as plt
from DoWnGAN.prep_gan import (
    load_data,
    mask_and_standardize,
    dt_index,
    find_nearest_index,
    to_utc,
    filter_times
)

from DoWnGAN.training import Trainer
from DoWnGAN.dataloader import NetCDFSR, xr_standardize_field
from DoWnGAN.utils import define_experiment, write_tags
from DoWnGAN.models.generator import Generator
from DoWnGAN.models.critic import Critic

torch.cuda.empty_cache()
dask.config.set({"array.slicing.split_large_chunks": True})

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def define_hyperparameters(pretrained_hash = None):

    hyper_params = {
        "gp_weight": 10,
        "critic_iterations": 5,
        "batch_size": 64,
        "gamma": 0.01,
        "content_weight": 5,
        "ncomp": 75,
        "lr": 0.00025,
    }

    run_params = {
        "epochs": 1000,
        "print_every": 250,
        "save_every": 250,
        "use_cuda": True,
        "device": device,
        "log_path": resource_filename("DoWnGAN", "mlflow_experiments"),
        "data_path": os.environ.get('DATA_PATH')
    }

    mlclient = MlflowClient(tracking_uri=run_params["log_path"])

    critic = Critic(16, 128, 2).to(device)
    generator = Generator(16, 128, 7).to(device)

    G_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=hyper_params["lr"],
        betas=(0.9, 0.99)
    )

    C_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=hyper_params["lr"],
        betas=(0.9, 0.99)
    )

    models = {
        "generator": generator,
        "critic": critic,
        "gen_optimizer": G_optimizer,
        "critic_optimizer": C_optimizer,
    }

    d = dict(
        epochs=run_params["epochs"],
        models=models,
        hyper_params=hyper_params,
        run_params=run_params,
        experiment=define_experiment(mlclient),
        tags=write_tags(),
        sf = 8
    )

    return d 


def run_it():
    """Commense training. Loads and processes the training data, partitions
    into train and validation sets 
    """
    pars = define_hyperparameters()

    fine = xr.open_dataset(os.environ.get('DATA_PATH')+f"/train_gt/{region}/wrf_{florida}_netcdf_train_years.nc")
    fine_v = xr.open_dataset(os.environ.get('DATA_PATH')+f"/validation_gt/{region}/wrf_{florida}_netcdf_validation_years.nc")
    
    fine_c, coarse_c = mask_and_standardize(
        anti_time_mask,
        data["fine_u"],
        data["fine_v"],
        coarse_u10,
        coarse_v10,
        pars["sf"]
    )

    randomized_validation = np.random.choice(fine_c.shape[0], 100)
    fine_c = fine_c[randomized_validation, ...]
    coarse_c = coarse_c[randomized_validation, ...]


    print("Masked and Standardized")

    # Free up some memory
    for var in [coarse_u10, coarse_v10, times, time_mask]:
        del var

    fine_t = torch.from_numpy(np.array(fine)).float()
    del fine

    fine_t_c = torch.from_numpy(np.array(fine_c)).float()
    del fine_c
    print("Stack fine")

    coarse_t = torch.from_numpy(np.array(coarse)).float()
    del coarse


    coarse_t_c = torch.from_numpy(np.array(coarse_c)).float()
    del coarse_c
    print("Stack coarse")

    dataset = NetCDFSR(fine_t, coarse_t, device=device)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=pars["hyper_params"]["batch_size"], shuffle=True
    )

    real_batch, real_cbatch = next(iter(dataloader))
    fixed = {
        "coarse": real_cbatch[:3, ...],
        "fine": real_batch[:3, ...],
    }

    validate = {
        "coarse": coarse_t_c,
        "fine": fine_t_c
    }

    for var in [fine_t, coarse_t]:
        del var

    trainer = Trainer(
        pars["models"],
        pars["hyper_params"],
        pars["run_params"],
        pars["experiment"],
        pars["tags"]
    )

    # Train model for N epochs
    trainer.train(
        dataloader, 
        epochs=pars["epochs"],
        fixed=fixed,
        validate=validate,
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    client = Client(n_workers=1, threads_per_worker=8, memory_limit="30GB")
    run_it()
