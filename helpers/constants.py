import os
from wrf_times import datetime_wrf_period
from datetime import datetime

if os.getenv('FINE_DATA_PATH_U10') is None:
    raise ValueError('Are env variables set? Use set_envs.sh for template')

# Consants
fine_paths_dict = {
    "u10": os.environ.get('FINE_DATA_PATH_U10'),
    "v10": os.environ.get('FINE_DATA_PATH_V10')
}
coarse_path = os.environ.get('COARSE_DATA_PATH')


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
    "u10": os.environ.get('COVARIATE_DATA_PATH')+"/interim_2000-10-01_to_2013-09-30.nc",
    "v10": os.environ.get('COVARIATE_DATA_PATH')+"/interim_2000-10-01_to_2013-09-30.nc",
    "land_sea_mask": os.environ.get('COVARIATE_DATA_PATH')+"/_grib2netcdf-webmars-public-svc-blue-005-6fe5cac1a363ec1525f54343b6cc9fd8-J3TMVF.nc",
    "surface_roughness": os.environ.get('COVARIATE_DATA_PATH')+"/interim_roughness_2000-10-01_to_2013-09-30.nc",
    "surface_pressure": os.environ.get('COVARIATE_DATA_PATH')+"/interim_surface_pressure_2000-10-01_to_2013-09-30.nc",
    "cape": os.environ.get('COVARIATE_DATA_PATH')+"/regrid_era5_era_cape_2000-10-01-2013-09-30.nc",
    "geopotential": os.environ.get('COVARIATE_DATA_PATH')+"/geopotential_era_interim.nc",
}

invariant_fields = ["land_sea_mask", "geopotential", "geopotential2"]
covariate_names_ordered = {
    # Standard name: variable name in netcdf
    "u10": 'u10',
    "v10":"v10",
    "land_sea_mask":"lsm",
    "surface_pressure":"sp",
    "surface_roughness":"sr",
    "geopotential":"z",
    "cape":"z"
}

fine_names_ordered = {"u10": "u10", "v10": "v10"}

# One of florida, west, or central
region = "florida"

regions = {
    "florida": {"lat_min": 4, "lat_max": 20, "lon_min": 70, "lon_max": 86},
    "central": {"lat_min": 30, "lat_max": 46, "lon_min": 50, "lon_max": 66},
    "west": {"lat_min": 30, "lat_max": 46, "lon_min": 15, "lon_max": 31},
}

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

# Compute constants
cpu_count = os.cpu_count()
chunk_size = 150

# I/O
proc_path = os.getenv('PROC_DATA')