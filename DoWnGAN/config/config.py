from DoWnGAN.helpers.wrf_times import datetime_wrf_period

import os
from datetime import datetime

import torch

# Path to HR data. Files are organized by variable and are loaded by xarray.open_mfdataset()
FINE_DATA_PATH_U10 = '/home/nannau/msc/Fall_2021/DoWnGAN/DoWnGAN/data/wrf/U10_regrid_16/regrid_16*.nc'
FINE_DATA_PATH_V10 = '/home/nannau/msc/Fall_2021/DoWnGAN/DoWnGAN/data/wrf/V10_regrid_16/regrid_16*.nc'
# Root dir for the covariates. Individual files defined below
COVARIATE_DATA_PATH = '/home/nannau/msc/Fall_2021/DoWnGAN/DoWnGAN/data'
# Where you want the processed data
PROC_DATA = '/home/nannau/deeplearning/production/data/proc_data'

# Where to store the mlflow tracking information. Make sure there is plenty of storage. 
# This repo is NOT conservative with IO.
EXPERIMENT_PATH = '/home/nannau/deeplearning/production/mlflow/mlflow_experiments'
EXPERIMENT_TAG="Configure production"

# Whether to load preprocessed data
already_preprocessed = True

# Which CUDA device to see
device = torch.device("cuda:0")

# One of florida, west, or central
# One of florida, west, or central
# region = "florida"
region = "florida"
invariant_fields = ["land_sea_mask", "geopotential"]

# Choose a reference field
ref_coarse = "u10"

# Masking years
mask_years = [2000, 2006, 2010]

# Scale factor for the covariates
scale_factor = 8

# WRF Time slice
# Add extra  6 hour step early due to peculiarities in WRF (extra field)
# Actual starting time of WRF is 2000-10-01 00:00:00
start_time = datetime(2000, 9, 30, 18, 0)
end_time = datetime(2013, 9, 30, 18, 0)
range_datetimes = datetime_wrf_period(start_time, end_time)

# Compute constants, machine dependent.
cpu_count = os.cpu_count()
chunk_size = 150


###########################################################
#### These are all of the options that can be configured###
###########################################################

"""
This file contains all of the configurable options that can be accessed for this project abd us exclusively dictionaries.
It uses paths defined in config.py
"""

# Variables in HR fields, and paths to those netcdfs
# This assumes they are separate files
fine_paths_dict = {
    "u10": FINE_DATA_PATH_U10,
    "v10": FINE_DATA_PATH_V10
}


# Rename attributes to a coherent naming convention
non_standard_attributes = {
    "latitude": "lat",
    "longitude": "lon",
    "Times": "time",
    "Time": "time",
    "times": "time",
    "U10": "u10",
    "V10": "v10",
}

# Covariate paths list
cov_paths_dict = {
    "u10": COVARIATE_DATA_PATH+"/interim_2000-10-01_to_2013-09-30.nc",
    "v10": COVARIATE_DATA_PATH+"/interim_2000-10-01_to_2013-09-30.nc",
    "land_sea_mask": COVARIATE_DATA_PATH+"/_grib2netcdf-webmars-public-svc-blue-005-6fe5cac1a363ec1525f54343b6cc9fd8-J3TMVF.nc",
    "surface_pressure": COVARIATE_DATA_PATH+"/interim_surface_pressure_2000-10-01_to_2013-09-30.nc",
    "surface_roughness": COVARIATE_DATA_PATH+"/interim_roughness_2000-10-01_to_2013-09-30.nc",
    "geopotential": COVARIATE_DATA_PATH+"/geopotential_era_interim.nc",
    "cape": COVARIATE_DATA_PATH+"/regrid_era5_era_cape_2000-10-01-2013-09-30.nc",
}

# Common names ordered, Just add variables into this dictionary when extending.
covariate_names_ordered = {
    # Standard name: variable name in netcdf
    "u10": 'u10',
    "v10": "v10",
    "land_sea_mask": "lsm",
    "surface_pressure": "sp",
    "surface_roughness": "sr",
    "geopotential": "z",
    "cape": "cape"
}

fine_names_ordered = {"u10": "u10", "v10": "v10"}


# These define the region indices in the coarse resolution
# They are multiplied by the scale factor to get them in the HR field.
# This assumes that the HR grids fit perfectly into the LR grids
regions = {
    "florida": {"lat_min": 4, "lat_max": 20, "lon_min": 70, "lon_max": 86},
    "central": {"lat_min": 30, "lat_max": 46, "lon_min": 50, "lon_max": 66},
    "central_larger": {"lat_min": 9, "lat_max": 47, "lon_min": 29, "lon_max": 67},
    "west": {"lat_min": 30, "lat_max": 46, "lon_min": 15, "lon_max": 31},
}

