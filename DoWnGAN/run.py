import gc

from dataloader import NetCDFSR, xr_standardize_field
from models.generator import Generator
from models.critic import Critic
from mlflow.tracking import MlflowClient
import mlflow

import pickle
import glob

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
from training import Trainer
from DoWnGAN.prep_gan import (
    load_data,
    mask_and_standardize,
    get_eofs_and_project,
    dt_index,
    find_nearest_index,
    to_utc,
    filter_times
)


torch.cuda.empty_cache()
dask.config.set({"array.slicing.split_large_chunks": True})

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def define_hyperparameters(pretrained_hash = None):

    hyper_params = {
        "gp_weight": 10,
        "critic_iterations": 5,
        "batch_size": 64,
        "gamma": 0.01,
        "eof_weight": 0.5,
        "div_weight": (1 - 0.01) / 4,
        "vort_weight": (1 - 0.01) / 4,
        "content_weight": 0.5,
        "ncomp": 75,
        "lr": 0.00025,
    }

    run_params = {
        "epochs": 500,
        "print_every": 250,
        "save_every": 250,
        "use_cuda": True,
        "device": device,
        "log_path": "mlruns",
        "load_eofs": False
    }

    if pretrained_hash is not None:
        logged_model_gen = f'file:///home/nannau/msc/DoWnGAN/mlruns/1/{pretrained_hash}/artifacts/Generator'
        logged_model_critic = f'file:///home/nannau/msc/DoWnGAN/mlruns/1/{pretrained_hash}/artifacts/Critic'

        # Load model as a PyFuncModel.
        generator = mlflow.pytorch.load_model(logged_model_gen).to(device)
        state_dict = mlflow.pytorch.load_state_dict(logged_model_gen)
        generator.load_state_dict(state_dict)

        critic = mlflow.pytorch.load_model(logged_model_critic).to(device)
        state_dict = mlflow.pytorch.load_state_dict(logged_model_critic)
        critic.load_state_dict(state_dict)

    else:
        critic = Critic(16, 128, 2).to(device)
        generator = Generator(16, 128, 2).to(device)

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
        experiment="WGAN-GP Extended Loss Equal Weights",
        tags={"description": "equal weights on loss"},
        sf = 8
    )

    return d 


def run_it():
    # torch.multiprocessing.freeze_support()

    # pars = define_hyperparameters(pretrained_hash="c8d5915d951547ee9c60da90a98ee1cc")
    pars = define_hyperparameters()

    fine_paths = {
        "U": "data/wrf/U10_regrid_16/regrid_16_6hrly_wrf2d_d01_ctrl_U10*.nc",
        "V": "data/wrf/V10_regrid_16/regrid_16_6hrly_wrf2d_d01_ctrl_V10*.nc"
    }

    coarse_paths = {
        "UV": "./data/interim_2000-10-01_to_2013-09-30.nc"
    }

    data = load_data(fine_paths, coarse_paths)

    coarse_u10 = data["coarse"].u10.loc["2000-01-01":"2015-05-30"]#.chunk({"time": 250})
    coarse_v10 = data["coarse"].v10.loc["2000-01-01":"2015-05-30"]#.chunk({"time": 250})

    # Extract times in datetime format
    times = dt_index(data["fine_u"].Times)

    # Apply filter to times for months you'd like
    time_mask = filter_times(times, mask_months=False, test_fraction=0.1)

    fine, coarse = mask_and_standardize(
        time_mask,
        data["fine_u"],
        data["fine_v"],
        coarse_u10,
        coarse_v10,
        pars["sf"]
    )

    print("Masked and Standardized")

    # Stochasticity
    # Random fine
    # rand_fine = xr.DataArray(np.random.uniform(-1, 1, u10.shape), coords=u10.coords)
    #Random coarse
    # rand_coarse = xr.DataArray(np.random.uniform(-1, 1, coarse_u10_patch.shape), coords=coarse_u10_patch.coords)

    # coarse = xr.concat([coarse_u10_patch, coarse_v10_patch], dim="var").transpose('time', 'var', 'latitude', 'longitude')
    # fine = xr.concat([u10, v10], dim="var").transpose('Times', 'var', 'lat', 'lon')

    # Free up some memory
    for var in [coarse_u10, coarse_v10, times, time_mask]:
        del var

    # Compute PCA


    # if not pars["run_params"]["load_eofs"]:
    # Weighted by explained variance
    EOFu, Zu, transformer_u = get_eofs_and_project(pars["hyper_params"]["ncomp"], fine[:, 0, ...])
    U = EOFu, Zu, transformer_u
    EOFv, Zv, transformer_v = get_eofs_and_project(pars["hyper_params"]["ncomp"], fine[:, 1, ...])
    V = EOFv, Zv, transformer_v
    print("Both EOFs fit")
    # with open(resource_filename("DoWnGAN", "data/eofs.pickle"), "wb") as f:
    #     pickle.dump((U, V), f)
    # print("Pickled!")
    # else:
    #     with open(resource_filename("DoWnGAN", "data/eofs.pickle"), "rb") as f:
    #         U, V = pickle.load(f)
    #         EOFu, Zu, transformer_u = U
    #         EOFv, Zv, transformer_v = V

    transformer = (transformer_u, transformer_v)

    EOFs = torch.from_numpy(np.stack([EOFu, EOFv], axis=1))
    for var in [EOFu, EOFv]:
        del var
    print("Stack EOFs")

    Z = torch.from_numpy(np.stack([Zu, Zv], axis=1)).float()
    print("Stack Z")
    for var in [Zu, Zv]:
        del var


    # u10_low = np.array([np.matmul(pca.components_.T, Zu[i, ...]).reshape(u10.shape[1], u10.shape[2]) for i in range(Zu.shape[0])])
    # v10_low = np.array([np.matmul(pca.components_.T, Zv[i, ...]).reshape(v10.shape[1], v10.shape[2]) for i in range(Zv.shape[0])])
    # low_uv10 = np.stack([u10_low, v10_low])


    # fine_t_low = torch.from_numpy(np.array(low_uv10))
    # del low_uv10
    # fine_t_high = fine_t - fine_t_low

    fine_t = torch.from_numpy(np.array(fine)).float()
    del fine
    print("Stack fine")

    coarse_t = torch.from_numpy(np.array(coarse)).float()
    del coarse
    print("Stack coarse")

    dataset = NetCDFSR(fine_t, coarse_t, Z, device=device)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=pars["hyper_params"]["batch_size"], shuffle=True
    )

    real_batch, real_cbatch, Zs = next(iter(dataloader))
    fixed = {
        "coarse": real_cbatch[:3, ...],
        "fine": real_batch[:3, ...],
        "Z": Zs[:3, ...]
    }

    for var in [fine_t, coarse_t, Z]:
        del var

    client = MlflowClient()
    print(client.get_experiment_by_name(pars["experiment"]).experiment_id)
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
        EOFs=EOFs,
        fixed=fixed,
        transformer = transformer
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    client = Client(n_workers=1, threads_per_worker=8, memory_limit="30GB")
    run_it()
