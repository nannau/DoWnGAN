import numpy as np
from mlflow import log_param
import datetime as datetime
import pandas as pd
import torch

import glob
import xarray as xr
# from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from pkg_resources import resource_filename

from DoWnGAN.dataloader import xr_standardize_field


def find_nearest_index(data, val):
    """Bisect search algorithm to find a value within a monotonically
    increasing array
    Args:
        data (np.ndarray): monotonically increasing array of column or row
            coordinates
        val (float): location of grid cell in x (rlon) or y (rlat) coords
    Returns:
        best_ind (integer): index in data of closest data value to val
    Raises:
        TypeError, ValueError in check_find_nearest_index_inputs
        TypeError:
                If data or val are not the correct type
        ValueError:
                If data is not monotonically increasing
                If size is not greater than 1
                If val is not within data's range of values
    """
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = int(lo + (hi - lo) / 2)
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind]
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind


def to_utc(d):
    ts = (d - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return ts

def dt_index(times):
    # Only works for WRF style time data
    Times_dt = []
    for t in times:
        year = int(str(float(t))[:4])
        month = int(str(float(t))[4:6])
        day = int(str(float(t))[6:8])
        hours = int(np.round(24 * float(str(float(t))[8:])))
        Times_dt.append(np.datetime64(datetime.datetime(year, month, day, hours)))
    Times_dt = np.array(Times_dt).astype("datetime64[D]")
    Times_dt_pd = pd.DatetimeIndex(Times_dt)

    return Times_dt_pd

def filter_times(times, mask_years=None):
    if mask_years is not None:
        print(f"Masking these years: {mask_years}")
        filter_func = np.vectorize(lambda x: True if x.year not in mask_years else False)
        time_mask = filter_func(times.astype(object))
        return time_mask

def mlflow_dict_logger(d: dict):
    for key in d.keys():
        log_param(key, d[key])


def load_data(fine_paths, coarse_paths):

    path_fine_u = glob.glob(fine_paths["U"])
    path_fine_v = glob.glob(fine_paths["V"])

    fine_u = xr.open_mfdataset(
            path_fine_u,
            combine="by_coords",
            engine="netcdf4",
            concat_dim="Time",
            # chunks={"Times": 250},
        )

    fine_v = xr.open_mfdataset(
        path_fine_v,
        combine="by_coords",
        engine="netcdf4",
        concat_dim="Time",
        # chunks={"Times": 250},
    )

    # Load ERA Interim
    coarse = xr.open_dataset(coarse_paths["UV"], engine="scipy").astype("float")
    # Organize lats in increasing order:
    coarse = coarse.sortby("latitude", ascending=True)

    return {
        "fine_u": fine_u,
        "fine_v": fine_v,
        "coarse": coarse
    }

def mask_and_standardize(time_mask, u10, v10, coarse_u10, coarse_v10, sf):
    
    coarse_constants = resource_filename("DoWnGAN", "data/_grib2netcdf-webmars-public-svc-blue-005-6fe5cac1a363ec1525f54343b6cc9fd8-J3TMVF.nc")
    coarse_sf = resource_filename("DoWnGAN", "data/interim_roughness_2000-10-01_to_2013-09-30.nc")
    coarse_sf = xr.open_dataset(coarse_sf)
    coarse_sf = coarse_sf.sortby('latitude', ascending=True)
    
    coarse_sp_path = resource_filename("DoWnGAN", "data/interim_surface_pressure_2000-10-01_to_2013-09-30.nc")
    coarse_sp = xr.open_dataset(coarse_sp_path)
    coarse_sp = coarse_sp.sortby("latitude", ascending=True)

    coarse_cape_path = resource_filename("DoWnGAN", "data/regrid_era5_era_cape_2000-10-01-2013-09-30.nc")
    coarse_cape = xr.open_dataset(coarse_cape_path)
    coarse_cape = coarse_cape.rename({"lat": "latitude", "lon": "longitude"})
    coarse_cape = coarse_cape.sortby("latitude", ascending=True).assign_coords({"time": coarse_u10.time, "latitude": coarse_u10.latitude, "longitude": coarse_u10.longitude})

    coarse_geo_path = resource_filename("DoWnGAN", "data/geopotential_era_interim.nc")
    coarse_geo = xr.open_dataset(coarse_geo_path)
    coarse_geo = coarse_geo.sortby("latitude", ascending=True)

    coarse_mask = xr.open_dataset(coarse_constants)
    coarse_mask = coarse_mask.sortby('latitude', ascending=True)
    mlat1 = np.argmin(np.abs(u10.lat.min()-coarse_mask.latitude).values)
    mlat2 = np.argmin(np.abs(u10.lat.max()-coarse_mask.latitude).values)
    mlon1 = np.argmin(np.abs(u10.lon.min()-(-360+coarse_mask.longitude)).values)
    mlon2 = np.argmin(np.abs(u10.lon.max()-(-360+coarse_mask.longitude)).values)+1
    coarse_mask = coarse_mask.lsm[0, mlat1:mlat2, mlon1:mlon2]

    list_mask = [coarse_mask for i in range(coarse_u10.time.shape[0])]
    coarse_mask = xr.concat(list_mask, dim="time").assign_coords({"time": coarse_u10.time, "latitude": coarse_u10.latitude, "longitude": coarse_u10.longitude})

    low, up, l, r = 4, 20, 70, 86 # Florida
    # low, up, l, r = 30, 46, 50, 66 # Central USA
    # low, up, l, r = 30, 46, 15, 31 # West coast

    list_geo = [coarse_geo.z[0, ...] for i in range(coarse_u10.time.shape[0])]
    coarse_geo = xr.concat(list_geo, dim="time").assign_coords({"time": coarse_u10.time, "latitude": coarse_u10.latitude, "longitude": coarse_u10.longitude})

    u10_patch = u10.U10[time_mask, sf * low : sf * up, sf * l : sf * r]
    v10_patch = v10.V10[time_mask, sf * low : sf * up, sf * l : sf * r]
    # fmask = np.repeat(fine_mask.LANDMASK.values[0, sf*low:sf*up, sf*l:sf*r][np.newaxis, :, :], u10.shape[0], axis=0)


    coarse_u10_patch = coarse_u10[time_mask, low:up, l:r]
    coarse_v10_patch = coarse_v10[time_mask, low:up, l:r]
    coarse_mask = coarse_mask[time_mask, low:up, l:r]
    coarse_sf = coarse_sf.sr[time_mask, low:up, l:r]
    coarse_sp = coarse_sp.sp[time_mask, low:up, l:r]
    coarse_cape = coarse_cape.cape[time_mask, low:up, l:r]
    coarse_geo = coarse_geo[time_mask, low:up, l:r]
    # cmask = np.repeat(coarse_mask.values[low:up, l:r][np.newaxis, :, :], u10.shape[0], axis=0)
    print(coarse_u10_patch.shape, coarse_v10_patch.shape, coarse_mask.shape, coarse_sp.shape, coarse_sf.shape, coarse_geo.shape, coarse_cape.shape)


    u10_patch = xr_standardize_field(u10_patch)
    v10_patch = xr_standardize_field(v10_patch)

    coarse_u10_patch = xr_standardize_field(coarse_u10_patch)
    coarse_v10_patch = xr_standardize_field(coarse_v10_patch)

    coarse_sp = xr_standardize_field(coarse_sp)
    coarse_sf = xr_standardize_field(coarse_sf)
    coarse_geo = xr_standardize_field(coarse_geo)
    coarse_cape = xr_standardize_field(coarse_cape)

    coarse = xr.concat([coarse_u10_patch, coarse_v10_patch, coarse_mask, coarse_sp, coarse_sf, coarse_geo, coarse_cape], dim="var").transpose('time', 'var', 'latitude', 'longitude')
    # coarse = xr.concat([coarse_u10_patch, coarse_v10_patch, coarse_mask, coarse_sp, coarse_sf], dim="var").transpose('time', 'var', 'latitude', 'longitude')
    # coarse = xr.concat([coarse_u10_patch, coarse_v10_patch], dim="var").transpose('time', 'var', 'latitude', 'longitude')

    fine = xr.concat([u10_patch, v10_patch], dim="var").transpose('Times', 'var', 'lat', 'lon')

    # coarse = np.stack([coarse_u10_patch, coarse_v10_patch], axis=1)
    # fine = np.stack([u10_patch, v10_patch], axis=1)

    assert fine.Times.shape == coarse.time.shape

    return (
        fine,
        coarse
    )


def mask_and_standardize_non_spatial_mask(time_mask, u10, v10, coarse_u10, coarse_v10, sf):

    low, up, l, r = 4, 20, 70, 86

    u10_patch = u10.U10[time_mask, sf * low : sf * up, sf * l : sf * r]
    v10_patch = v10.V10[time_mask, sf * low : sf * up, sf * l : sf * r]
    # fmask = np.repeat(fine_mask.LANDMASK.values[0, sf*low:sf*up, sf*l:sf*r][np.newaxis, :, :], u10.shape[0], axis=0)

    coarse_u10_patch = coarse_u10[time_mask, low:up, l:r]
    coarse_v10_patch = coarse_v10[time_mask, low:up, l:r]
    # cmask = np.repeat(coarse_mask.values[low:up, l:r][np.newaxis, :, :], u10.shape[0], axis=0)

    u10_patch = xr_standardize_field(u10_patch)
    v10_patch = xr_standardize_field(v10_patch)

    coarse_u10_patch = xr_standardize_field(coarse_u10_patch)
    coarse_v10_patch = xr_standardize_field(coarse_v10_patch)

    coarse = xr.concat([coarse_u10_patch, coarse_v10_patch], dim="var").transpose('time', 'var', 'latitude', 'longitude')
    fine = xr.concat([u10_patch, v10_patch], dim="var").transpose('Times', 'var', 'lat', 'lon')

    # coarse = np.stack([coarse_u10_patch, coarse_v10_patch], axis=1)
    # fine = np.stack([u10_patch, v10_patch], axis=1)


    return (
        fine,
        coarse
    )


def get_eofs_and_project(ncomp, X):

    bsize, nlat, nlon = X.shape
    pca = PCA(ncomp)
    # pca = IncrementalPCA(ncomp, batch_size=5)

    # for chunk in np.array_split(X, 50, axis=0):
    #     pca.partial_fit(
    #         np.array(
    #             chunk
    #         ).reshape(
    #                 chunk.shape[0],
    #                 nlat * nlon
    #         )
    #     )

    pca.fit(np.array(X).reshape(X.shape[0], nlat * nlon))

    # Weight by explained variance
    EOFs = pca.components_#/pca.explained_variance_[:, np.newaxis]
    # Project onto EOFs/perform dimensionality reduction
    Z = pca.transform(
            np.array(
                X
            ).reshape(
                    bsize,
                    nlat * nlon
            )
        )

    return EOFs, Z, pca