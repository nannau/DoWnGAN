# Generates experiment netcdf files
import glob
from xarray.core import dataset
from xarray.core.dataset import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from helpers.wrf_times import filter_times, wrf_to_dt
from dask.distributed import Client, LocalCluster
import dask

import matplotlib.pyplot as plt

import helpers.constants as c

dask.config.set({"array.slicing.split_large_chunks": True})


def crop_dataset(ds: xr.Dataset, scale_factor: int) -> xr.Dataset:
    """
    Crops the dataset to the region of interest.
    """
    lat1, lat2 = c.regions[c.region]["lat_min"], c.regions[c.region]["lat_max"]
    lon1, lon2 = c.regions[c.region]["lon_min"], c.regions[c.region]["lon_max"]

    lat1, lat2, lon1, lon2 = (lat1*scale_factor, lat2*scale_factor, lon1*scale_factor, lon2*scale_factor)

    if isinstance(ds, xr.Dataset):
        for var in list(ds.data_vars):
            cropped_ds = ds[var][:, lat1:lat2, lon1:lon2]
        return cropped_ds

    return ds[:, lat1:lat2, lon1:lon2]


def standardize_attribute_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardizes the attribute names of the dataset.
    """
    keylist = list(ds.keys()) + list(ds.coords)
    for key in keylist:
        if key in c.non_standard_attributes.keys():
            print(f"Renamed {key} -> {c.non_standard_attributes[key]}")
            ds = ds.rename({key: c.non_standard_attributes[key]})

    return ds


def extend_along_time(da: xr.DataArray) -> xr.DataArray:
    """
    Extends the data array along the time dimension to match
    a reference dataset for time invariant fields.
    """
    print("Extending on the time dimension...")
    list_times = [da for _ in c.range_datetimes]
    da_ext = xr.concat(list_times, dim="time").assign_coords({"time": c.range_datetimes})

    return da_ext

def load_fine(path_dict: dict) -> dict:
    """
    Loads fine/wrf scale data from netcdf files. Assumes the data is
    stored in multiple netcdf files.
    
    Parameters:
    ----------- 
    path_dict: dict dictionary containing the paths to the multiple
        netcdf files.
    Returns:
    --------
    fine_dict: dict dictionary containing the dataset objects of
        the loaded data.
    """
    datasets_dict = {}
    # Load fine data
    for key in path_dict.keys():
        print("Opening: ", path_dict[key])
        if "*" in path_dict[key]:
            datasets_dict[key] = xr.open_mfdataset(
                glob.glob(path_dict[key]), 
                combine = "by_coords",
                engine = "netcdf4",
                parallel = True
            )
        else:
            datasets_dict[key] = xr.open_dataset(
                path_dict[key], 
                engine = "netcdf4",
            )
        # Standardize the dimension names so that
        # They're all the same!
        datasets_dict[key] = standardize_attribute_names(datasets_dict[key])
        datasets_dict[key] = crop_dataset(datasets_dict[key], c.scale_factor)

        print("Dataset dimensions ", datasets_dict[key].dims)
        datasets_dict[key]["time"] = wrf_to_dt(datasets_dict[key]["time"])

    return datasets_dict


def crop_global_mask(mask, ref_ds):
    """The saved mask is a global mask. This function masks the data
    with the local mask defined by the subdomain.
    """
    varname, = mask.data_vars
    mlat1 = np.argmin(np.abs(ref_ds.lat.min()-mask.lat).values)
    mlat2 = np.argmin(np.abs(ref_ds.lat.max()-mask.lat).values)
    mlon1 = np.argmin(np.abs(ref_ds.lon.min()-(-360+mask.lon)).values)
    mlon2 = np.argmin(np.abs(ref_ds.lon.max()-(-360+mask.lon)).values)+1
    mask = mask[varname][0, mlat1:mlat2, mlon1:mlon2]

    return mask

def load_covariates(path_dict: dict, ref_dataset: xr.Dataset) -> dict:
    """
    Loads covariates from netcdf files. 
    Parameters:
    -----------):
    """

    datasets_dict = {}
    # Load covariates
    for key in path_dict:
        print("Adding ", key)
        print("--"*80)
        ds = xr.open_dataset(path_dict[key], engine="netcdf4")
        # if isinstance(ds, xr.DataArray):
        ds = standardize_attribute_names(ds)

        # Additional preprocessing steps - assure that the data is sorted
        # by latitude
        ds = ds.sortby("lat", ascending=True)
        datasets_dict[key] = ds[c.covariate_names_ordered[key]]

        # Extend the data along the time dimension if invariant
        if key == "mask":
            datasets_dict[key] = crop_global_mask(datasets_dict[key], ref_dataset)

        datasets_dict[key] = crop_dataset(datasets_dict[key], 1)

        if key in c.invariant_fields:
            datasets_dict[key] = extend_along_time(datasets_dict[key])

    ref_coarse = datasets_dict[c.ref_coarse]
    for key in datasets_dict:
        datasets_dict[key] = datasets_dict[key].assign_coords({"time": c.range_datetimes, "lat": ref_coarse.lat, "lon": ref_coarse.lon})

    return datasets_dict


def concat_data_arrays(data_dict: dict, variable_order: list) -> xr.DataArray:
    """
    Concatenates a list of data arrays along the time dimension.
    """
    print("Order in processed dataset: ", variable_order.keys())
    ds = xr.Dataset()
    for var, key in zip(variable_order, data_dict):
        ds[var] = data_dict[key]

    print(80*"-")

    return ds


def train_test_split(coarse: xr.Dataset, fine: xr.Dataset) -> xr.Dataset:
    """Splits the data into train and test sets.
    """
    assert coarse.time.shape[0] == fine.time.shape[0], "Time dim on coarse and fine datasets do not match!"
    time_arr = fine.time
    train_time_mask = filter_times(time_arr, mask_years=c.mask_years)
    test_time_mask = ~train_time_mask.copy()

    # Mask out the first element from the year 2000 because its
    # an incorrect field
    if 2000 in c.mask_years:
        test_time_mask[0] = False

    coarse_train = coarse.loc[{"time": train_time_mask}]
    fine_train = fine.loc[{"time": train_time_mask}]

    coarse_test = coarse.loc[{"time": test_time_mask}]
    fine_test = fine.loc[{"time": test_time_mask}]

    assert coarse_train.time.shape[0] == fine_train.time.shape[0], "Train time dim on coarse and fine datasets do not match!"
    assert coarse_train.time.shape[0] == fine_train.time.shape[0], "Train time dim on coarse and fine datasets do not match!"
    assert coarse_test.time.shape[0] == fine_test.time.shape[0], "Test time dim on coarse and fine datasets do not match!"
    assert coarse_train.time.shape[0] == fine_train.time.shape[0], "Test time dim on coarse and fine datasets do not match!"

    return coarse_train, fine_train, coarse_test, fine_test


def xr_standardize_array(da: xr.DataArray) -> xr.DataArray:
    """Standardizes the data array to have mean of 0
    and standard deviation 1.
    """
    mean = da.mean(skipna=True)
    std = da.std(skipna=True)
    return (da - mean) / std

def xr_standardize_all(data_dict: dict) -> dict:
    """
    Standardizes the data arrays in the dictionary.
    """
    for key in data_dict:
        # Binary land mask does not need normalization
        if key != "land_sea_mask":
            std = float(data_dict[key].std())
            mean = float(data_dict[key].mean())
            print("-"*80)
            print(f"Before Mean of {key}", mean)
            print(f"Before Standard Deviation of {key}", std)

            data_dict[key] = xr_standardize_array(data_dict[key])
            std = float(data_dict[key].std())
            mean = float(data_dict[key].mean())
            print("-"*80)
            print(f"Mean of {key}", mean)
            print(f"Standard Deviation of {key}", std)
            # Surface pressure has such large units that the correction
            # is within a few orders of magnitude.
            if key != "surface_pressure":
                assert np.isclose(mean, 0., atol=1e-2), "Mean of the data is not 0!"
                assert np.isclose(std, 1., atol=1e-1), "Standard deviation of the data is not 1!"

            if key == "surface_pressure":
                assert np.isclose(mean, 0., atol=1e-2), "Mean of the data is not 0!"
                assert np.isclose(std, 1., atol=1.0), "Standard deviation of the data is not within tolerance!"


    return data_dict


def generate_train_test_coarse_fine():
    coarse_path = c.coarse_path
    cov_paths_dict = c.cov_paths_dict

    fine_xr_dict = load_fine(c.fine_paths_dict)
    fine_xr_dict = xr_standardize_all(fine_xr_dict)
    fine = concat_data_arrays(fine_xr_dict, c.fine_names_ordered)

    coarse_xr_dict = load_covariates(cov_paths_dict, fine)
    coarse_xr_dict = xr_standardize_all(coarse_xr_dict)
    # Chooese reference dataset to define lat and lon
    coarse = concat_data_arrays(coarse_xr_dict, c.covariate_names_ordered)

    # Train test split!
    coarse_train, fine_train, coarse_test, fine_test = train_test_split(coarse, fine)


    print("Final train set size:")
    print("Coarse")
    print("-"*80)
    print(coarse_train.head())
    print("Fine")
    print("-"*80)
    print(fine_train.head())
    print("Final test set size:")
    print("Coarse")
    print("-"*80)
    print(coarse_test.head())
    print("Fine")
    print("-"*80)
    print(fine_test.head())

    return coarse_train, fine_train, coarse_test, fine_test


def load_preprocessed():
    coarse_train = xr.open_dataset(c.proc_path+f"/coarse_train_{c.region}.nc", engine="netcdf4")
    fine_train = xr.open_dataset(c.proc_path+f"/fine_train_{c.region}.nc", engine="netcdf4")
    coarse_test = xr.open_dataset(c.proc_path+f"/coarse_test_{c.region}.nc", engine="netcdf4")
    fine_test = xr.open_dataset(c.proc_path+f"/fine_test_{c.region}.nc", engine="netcdf4")

    return coarse_train, fine_train, coarse_test, fine_test


