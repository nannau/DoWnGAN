from dataloader import NetCDFSR, xr_standardize_field
import xarray as xr
import numpy as np
import pandas as pd
import glob
import torch
from prep_gan import find_nearest_index, to_utc

from distributed import Client
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA
from scipy.interpolate import NearestNDInterpolator

import matplotlib.pyplot as plt
from models.generator import Generator
from models.critic import Critic


torch.cuda.empty_cache()
import dask


def run_it():
    dask.config.set({"array.slicing.split_large_chunks": True})

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Depending on your workstation specifications, you may need to adjust these values.
    # On a single machine, n_workers=1 is usually better.
    client = Client(n_workers=1, threads_per_worker=8, memory_limit="15GB")

    hr_dataroot_U10 = "data/wrf/U10_regrid_16/regrid_16_6hrly_wrf2d_d01_ctrl_U10*.nc"
    hr_dataroot_V10 = "data/wrf/V10_regrid_16/regrid_16_6hrly_wrf2d_d01_ctrl_V10*.nc"

    fine_u = xr.open_mfdataset(
        glob.glob(hr_dataroot_U10),
        combine="by_coords",
        engine="netcdf4",
        concat_dim="Time",
        chunks={"Times": 100},
    )
    fine_v = xr.open_mfdataset(
        glob.glob(hr_dataroot_V10),
        combine="by_coords",
        engine="netcdf4",
        concat_dim="Time",
        chunks={"Times": 100},
    )

    dataroot = "./data/interim_2000-10-01_to_2013-09-30.nc"

    target = "10UV_GDS4_SFC"
    coarse = xr.open_dataset(dataroot).astype("float")
    coarse = coarse.sortby("latitude", ascending=True)
    coarse_u10 = coarse.u10.loc["2000-01-01":"2015-05-30"].chunk({"time": 100})
    coarse_v10 = coarse.v10.loc["2000-01-01":"2015-05-30"].chunk({"time": 100})

    # scale factor
    sf = 8

    tidx = 15000
    vmin, vmax = -13, 6
    low, up, l, r = 4, 20, 70, 86
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(
        fine_u.U10[tidx, sf * low : sf * up, sf * l : sf * r],
        origin="lower",
    )  # vmin=vmin, vmax=vmax)
    ax[1].imshow(
        coarse_u10[tidx, low:up, l:r],
        origin="lower",
    )  # vmin=vmin, vmax=vmax)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(
        fine_v.V10[tidx, sf * low : sf * up, sf * l : sf * r],
        origin="lower",
    )  # vmin=vmin, vmax=vmax)
    ax[1].imshow(
        coarse_v10[tidx, low:up, l:r],
        origin="lower",
    )  # vmin=vmin, vmax=vmax)

    import datetime as datetime

    Times_dt = []
    for t in fine_u.Times:
        year = int(str(float(t))[:4])
        month = int(str(float(t))[4:6])
        day = int(str(float(t))[6:8])
        hours = int(np.round(24 * float(str(float(t))[8:])))
        Times_dt.append(np.datetime64(datetime.datetime(year, month, day, hours)))
    Times_dt = np.array(Times_dt).astype("datetime64[D]")
    Times_dt_pd = pd.DatetimeIndex(Times_dt)

    filter_func = np.vectorize(lambda x: True if x.month in [6, 7, 8, 9, 10] else False)
    time_mask = filter_func(Times_dt_pd.astype(object))
    time_mask = np.ones(Times_dt.shape[0]) != 1
    time_mask[1:18000] = True

    u10 = fine_u.U10[time_mask, sf * low : sf * up, sf * l : sf * r]
    v10 = fine_v.V10[time_mask, sf * low : sf * up, sf * l : sf * r]
    # fmask = np.repeat(fine_mask.LANDMASK.values[0, sf*low:sf*up, sf*l:sf*r][np.newaxis, :, :], u10.shape[0], axis=0)

    coarse_u10_patch = coarse_u10[time_mask, low:up, l:r]
    coarse_v10_patch = coarse_v10[time_mask, low:up, l:r]
    # cmask = np.repeat(coarse_mask.values[low:up, l:r][np.newaxis, :, :], u10.shape[0], axis=0)

    u10 = xr_standardize_field(u10)
    v10 = xr_standardize_field(v10)
    coarse_u10_patch = xr_standardize_field(coarse_u10_patch)
    coarse_v10_patch = xr_standardize_field(coarse_v10_patch)

    # #Random fine
    # rand_fine = xr.DataArray(np.random.uniform(-1, 1, u10.shape), coords=u10.coords)
    # #Random coarse
    # rand_coarse = xr.DataArray(np.random.uniform(-1, 1, coarse_u10_patch.shape), coords=coarse_u10_patch.coords)
    coarse = np.stack([coarse_u10_patch, coarse_v10_patch], axis=1)
    fine = np.stack([u10, v10], axis=1)

    # coarse = xr.concat([coarse_u10_patch, coarse_v10_patch], dim="var").transpose('time', 'var', 'latitude', 'longitude')
    # fine = xr.concat([u10, v10], dim="var").transpose('Times', 'var', 'lat', 'lon')

    del coarse_u10_patch
    del coarse_v10_patch

    # PCA
    ncomp = 25
    fine_pca_u10 = np.array(fine[:, 0, ...]).reshape(
        u10.shape[0], u10.shape[1] * u10.shape[2]
    )
    pca = PCA(n_components=ncomp)
    pca.fit(fine_pca_u10)
    fine_sp_basis_u10 = (
        pca.components_.reshape(ncomp, u10.shape[1] * u10.shape[2])
        / pca.explained_variance_[:, np.newaxis]
    )
    Zu = pca.transform(fine_pca_u10)
    # Zu = pca.transform(Xu)
    # u10_low = np.array([np.matmul(pca.components_.T, Zu[i, ...]).reshape(u10.shape[1], u10.shape[2]) for i in range(Zu.shape[0])])
    pcau_comp = pca.components_

    del fine_pca_u10
    # fine_sp_basis_u10 = np.divide(fine_sp_basis_u10, pca.explained_variance_)

    fine_pca_v10 = np.array(fine[:, 1, ...]).reshape(
        v10.shape[0], v10.shape[1] * v10.shape[2]
    )
    pca = PCA(n_components=ncomp)
    pca.fit(fine_pca_v10)
    Zv = pca.transform(fine_pca_v10)
    # Zv = pca.transform(Xv)
    # v10_low = np.array([np.matmul(pca.components_.T, Zv[i, ...]).reshape(v10.shape[1], v10.shape[2]) for i in range(Zv.shape[0])])
    pcav_comp = pca.components_
    del fine_pca_v10

    fine_sp_basis_v10 = (
        pca.components_.reshape(ncomp, v10.shape[1] * v10.shape[2])
        / pca.explained_variance_[:, np.newaxis]
    )
    fine_sp_basis = np.stack([fine_sp_basis_u10, fine_sp_basis_v10], axis=1)
    # low_uv10 = np.stack([u10_low, v10_low])
    del fine_sp_basis_v10
    del fine_sp_basis_u10

    Zstack = np.stack([Zu, Zv], axis=1)

    fine_t = torch.from_numpy(np.array(fine)).float()
    # fine_t_low = torch.from_numpy(np.array(low_uv10))
    # del low_uv10
    # fine_t_high = fine_t - fine_t_low

    coarse_t = torch.from_numpy(np.array(coarse)).float()
    pcas_t = torch.from_numpy(fine_sp_basis).float()
    pca_og_shape = torch.from_numpy(np.stack([pcau_comp, pcav_comp], axis=1))
    Z_t = torch.from_numpy(Zstack).float()

    del u10
    del v10
    del fine_sp_basis
    del coarse
    del fine
    del Zstack

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    batch_size = 64
    dataset = NetCDFSR(fine_t, coarse_t, pcas_t, pca_og_shape, Z_t, device=device)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )

    real_batch, real_cbatch, pcas, pcas_og_shape, Zs = next(iter(dataloader))
    fixed = {"coarse": real_cbatch[:2, ...], "fine": real_batch[:2, ...]}

    critic = Critic(16, 128, 2).to(device)
    generator = Generator(16, 128, 2).to(device)
    lr = 0.00025

    import gc

    gc.collect()

    G_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99))
    C_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.9, 0.99))

    # Set up trainer
    from training import Trainer

    epochs = 500
    models = {
        "generator": generator,
        "critic": critic,
        "gen_optimizer": G_optimizer,
        "critic_optimizer": C_optimizer,
    }
    hyper_params = {
        "gp_weight": 10,
        "critic_iterations": 5,
        "batch_size": batch_size,
        "gamma": 0.01,
        "eof_weight": 0.5,
        "div_weight": (1 - 0.01) / 4,
        "vort_weight": (1 - 0.01) / 4,
        "content_weight": 0.5,
    }

    run_params = {
        "epochs": epochs,
        "print_every": 500,
        "save_every": 250,
        "use_cuda": True,
        "device": device,
        "log_path": "mlruns",
    }

    # trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
    #                   use_cuda=torch.cuda.is_available(), print_every=500, save_every=500, device=device)
    experiment = "WGAN-GP Extended Loss Equal Weights"
    tags = {"description": "equal weights on loss"}

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    # print(dir(client))
    print(client.get_experiment_by_name(experiment).experiment_id)

    trainer = Trainer(models, hyper_params, run_params, experiment, tags)

    # Train model for N epochs
    trainer.train(
        dataloader,
        epochs=epochs,
        fixed=fixed,
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_it()
