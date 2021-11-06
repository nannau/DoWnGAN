from DoWnGAN.prep_gan import (
    load_data,
    mask_and_standardize,
    dt_index,
    filter_times
)
from DoWnGAN.dataloader import xr_standardize_field
import click
import logging
import glob
import os
import warnings
import numpy as np
import xarray as xr
import torch
import mlflow

from pkg_resources import resource_filename
from dask.distributed import Client
from pkg_resources import resource_filename

warnings.filterwarnings("ignore")

region_area = {
    "florida": (4, 20, 70, 86),
    "central": (30, 46, 50, 66),
    "west": (30, 46, 15, 31)
}


@click.command()
@click.option("-r", "--region", help="One of west, central, florida", required=True)
@click.option("-s", "--set", help="One of validation or train sets", required=True)
@click.option(
    "-l",
    "--log-level",
    help="Logging level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO"
)
def main(region, set, log_level, sf=8):

    low, up, l, r = region_area[region]
    fine_paths = {
        "U": os.environ.get("NETCDF_PATH")+"data/wrf/U10_regrid_16/regrid_16_6hrly_wrf2d_d01_ctrl_U10*.nc"
    }
    coarse_paths = {
        "UV": os.environ.get("NETCDF_PATH")+"data/interim_2000-10-01_to_2013-09-30.nc"
    }

    logging.info("Loading data")
    u10 = xr.open_mfdataset(glob.glob(fine_paths["U"])).U10
    coarse = xr.open_dataset(coarse_paths["UV"])
    coarse_u10 = coarse.u10

    # Extract times in datetime format
    times = dt_index(u10.Times)

    # Apply filter to times for months you'd like
    time_mask = filter_times(times, mask_years=[2000, 2006, 2010])

    if set == "validation":
        time_mask = ~time_mask
    time_mask[0] = False

    coarse_sf = os.environ.get("DATA_PATH")+"/covariates/original_files/interim_roughness_2000-10-01_to_2013-09-30.nc"
    coarse_sf = xr.open_dataset(coarse_sf)
    coarse_sf = coarse_sf.sortby('latitude', ascending=True)
    
    coarse_sp_path = os.environ.get("DATA_PATH")+"/covariates/original_files/interim_surface_pressure_2000-10-01_to_2013-09-30.nc"
    coarse_sp = xr.open_dataset(coarse_sp_path)
    coarse_sp = coarse_sp.sortby("latitude", ascending=True)

    coarse_cape_path = os.environ.get("DATA_PATH")+"/covariates/original_files/regrid_era5_era_cape_2000-10-01-2013-09-30.nc"
    coarse_cape = xr.open_dataset(coarse_cape_path)
    coarse_cape = coarse_cape.rename({"lat": "latitude", "lon": "longitude"})
    coarse_cape = coarse_cape.sortby("latitude", ascending=True).assign_coords({"time": times, "latitude": coarse_u10.latitude, "longitude": coarse_u10.longitude})

    coarse_geo_path = os.environ.get("DATA_PATH")+"/covariates/original_files/geopotential_era_interim.nc"
    coarse_geo = xr.open_dataset(coarse_geo_path)
    coarse_geo = coarse_geo.sortby("latitude", ascending=True)

    coarse_constants = os.environ.get("DATA_PATH")+"/covariates/original_files/_grib2netcdf-webmars-public-svc-blue-005-6fe5cac1a363ec1525f54343b6cc9fd8-J3TMVF.nc"
    coarse_mask = xr.open_dataset(coarse_constants)
    coarse_mask = coarse_mask.sortby('latitude', ascending=True)

    logging.info("Masking relevant data")
    mlat1 = np.argmin(np.abs(u10.lat.min()-coarse_mask.latitude).values)
    mlat2 = np.argmin(np.abs(u10.lat.max()-coarse_mask.latitude).values)
    mlon1 = np.argmin(np.abs(u10.lon.min()-(-360+coarse_mask.longitude)).values)
    mlon2 = np.argmin(np.abs(u10.lon.max()-(-360+coarse_mask.longitude)).values)+1

    # define mask array
    coarse_mask = coarse_mask.lsm[0, mlat1:mlat2, mlon1:mlon2]

    # Generate list of fields/data arrays and concatenate along time (this is for time invariant fields)
    list_mask = [coarse_mask for i in range(times.shape[0])]
    coarse_mask = xr.concat(list_mask, dim="time").assign_coords({"time": times, "latitude": coarse_u10.latitude, "longitude": coarse_u10.longitude})

    # Same as mask
    list_geo = [coarse_geo.z[0, ...] for i in range(times.shape[0])]
    coarse_geo = xr.concat(list_geo, dim="time").assign_coords({"time": times, "latitude": coarse_u10.latitude, "longitude": coarse_u10.longitude})

    if set == "train":
        # Apply regional mask
        coarse_mask = coarse_mask[time_mask, low:up, l:r]
        coarse_sf = coarse_sf.sr[time_mask, low:up, l:r]
        coarse_sp = coarse_sp.sp[time_mask, low:up, l:r]
        coarse_cape = coarse_cape.cape[time_mask, low:up, l:r]
        coarse_geo = coarse_geo[time_mask, low:up, l:r]

        logging.info("Standardizing data")
        coarse_sp = xr_standardize_field(coarse_sp, coarse_sp.mean(), coarse_sp.std())
        coarse_sf = xr_standardize_field(coarse_sf, coarse_sf.mean(), coarse_sf.std())
        coarse_geo = xr_standardize_field(coarse_geo, coarse_geo.mean(), coarse_geo.std())
        coarse_cape = xr_standardize_field(coarse_cape, coarse_cape.mean(), coarse_cape.std())

    if set == "validation":
        # Need to calculate the total pixel statistics to standardize
        # But only use the statistics from the training set (important!)

        # Apply regional mask
        coarse_mask = coarse_mask[:, low:up, l:r]
        coarse_sf = coarse_sf.sr[:, low:up, l:r]
        coarse_sp = coarse_sp.sp[:, low:up, l:r]
        coarse_cape = coarse_cape.cape[:, low:up, l:r]
        coarse_geo = coarse_geo[:, low:up, l:r]

        # Apply regional mask (train)
        coarse_mask_train = coarse_mask[~time_mask, ...]
        coarse_sf_train = coarse_sf[~time_mask, ...]
        coarse_sp_train = coarse_sp[~time_mask, ...]
        coarse_cape_train = coarse_cape[~time_mask, ...]
        coarse_geo_train = coarse_geo[~time_mask, ...]

        # Apply regional mask
        coarse_mask = coarse_mask[time_mask, ...]
        coarse_sf = coarse_sf[time_mask, ...]
        coarse_sp = coarse_sp[time_mask, ...]
        coarse_cape = coarse_cape[time_mask, ...]
        coarse_geo = coarse_geo[time_mask, ...]

        logging.info("Standardizing data")
        coarse_sp = xr_standardize_field(coarse_sp, coarse_sp_train.mean(), coarse_sp_train.std())
        coarse_sf = xr_standardize_field(coarse_sf, coarse_sf_train.mean(), coarse_sf_train.std())
        coarse_geo = xr_standardize_field(coarse_geo, coarse_geo_train.mean(), coarse_geo_train.std())
        coarse_cape = xr_standardize_field(coarse_cape, coarse_cape_train.mean(), coarse_cape_train.std())

    logging.info("Write to file")
    coarse_sp.to_netcdf(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_surface_pressure.nc")
    coarse_mask.to_netcdf(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_land_sea_mask.nc")
    coarse_sf.to_netcdf(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_surface_friction.nc")
    coarse_cape.to_netcdf(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_cape.nc")
    coarse_geo.to_netcdf(os.environ.get("DATA_PATH")+f"/covariates/{set}/{region}_geopotential_height.nc")
    logging.info("Completed!")

if __name__ == "__main__":
    main()

