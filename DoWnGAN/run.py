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
from xarray.core import variable
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


def get_coarse_tensor(region, set):

    coarse_path = os.environ.get("DATA_PATH")+f"/{set}_gt/{region}/era_{region}_netcdf_{set}_years.nc"
    coarse = xr.open_dataset(coarse_path).to_array().transpose("time", "variable", "lat", "lon").rename({"lat":"latitude", "lon":"longitude"})

    coarse_mask = xr.open_dataset(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_land_sea_mask.nc").to_array().transpose("time", "variable", "latitude", "longitude")
    coarse_sf = xr.open_dataset(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_surface_friction.nc").to_array().transpose("time", "variable", "latitude", "longitude")
    coarse_geo = xr.open_dataset(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_geopotential_height.nc").to_array().transpose("time", "variable", "latitude", "longitude")
    coarse_sp = xr.open_dataset(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_surface_pressure.nc").to_array().transpose("time", "variable", "latitude", "longitude")
    coarse_cape = xr.open_dataset(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_cape.nc").to_array().transpose("time", "variable", "latitude", "longitude")


    fields = [
        coarse[:, 0,...].values,
        coarse[:, 1,...].values,
        coarse_mask.transpose("time", "variable", "latitude", "longitude")[:, 0, ...].values,
        coarse_sp[:, 0, ...].values,
        coarse_sf[:, 0, ...].values,
        coarse_geo[:, 0, ...].values,
        coarse_cape[:, 0, ...].values
    ]

    for i, cov in enumerate(fields):
        fields[i] = torch.from_numpy(cov)
        print(cov.shape, fields[0].shape)
        assert cov.shape == fields[0].shape

    coarse_t = torch.stack(fields, dim=1)
    return coarse_t

def run_it():
    """Commense training. Loads and processes the training data, partitions
    into train and validation sets 
    """
    region = "florida"

    pars = define_hyperparameters()

    coarse_t = get_coarse_tensor(region, "train")
    coarse_t_v = get_coarse_tensor(region, "validation")

    fine_path = os.environ.get('DATA_PATH')+f"/train_gt/{region}/wrf_{region}_netcdf_train_years.nc"
    fine = xr.open_dataset(fine_path).to_array().transpose("time", "variable", "lat", "lon")

    # Validation data

    fine_path_v = os.environ.get("DATA_PATH")+f"/validation_gt/{region}/wrf_{region}_netcdf_validation_years.nc"
    fine_v = xr.open_dataset(fine_path_v).to_array().transpose("time", "variable", "lat", "lon")

    randomized_validation = np.random.choice(fine_v.time.shape[0], 100)

    fine_v = fine_v[randomized_validation, ...]
    coarse_t_v = coarse_t_v[randomized_validation, ...]


    print("Masked and Standardized")


    fine_t = torch.from_numpy(np.array(fine)).float()
    del fine

    fine_t_v = torch.from_numpy(np.array(fine_v)).float()
    del fine_v
    print("Stack fine")

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
        "coarse": coarse_t_v,
        "fine": fine_t_v
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
