from DoWnGAN.prep_gan import (
    load_data,
    mask_and_standardize,
    dt_index,
    filter_times
)
from DoWnGAN.dataloader import xr_standardize_field
import glob
import click
import logging
import warnings
import numpy as np
import xarray as xr
import torch
import mlflow

from pkg_resources import resource_filename
from dask.distributed import Client
from pkg_resources import resource_filename

warnings.filterwarnings("ignore")


@click.command()
@click.option("-h", "--hash-code", help="Input hash from mlflow", required=True)
@click.option("-r", "--region", help="Region slice indices defined in coarse resolution", required=True)
@click.option("-e", "--epoch-number", help="Epoch number to load", required=True, default=999)
@click.option(
    "-l",
    "--log-level",
    help="Logging level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO"
)
def main(hash_code, region, epoch_number, log_level):
    # client = Client(n_workers=6, threads_per_worker=1, memory_limit="16GB")

    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    logging.basicConfig(level=log_level)

    fine_paths = {
        "U": resource_filename("DoWnGAN", "data/wrf/U10_regrid_16/regrid_16_6hrly_wrf2d_d01_ctrl_U10*.nc"),
        "V": resource_filename("DoWnGAN", "data/wrf/V10_regrid_16/regrid_16_6hrly_wrf2d_d01_ctrl_V10*.nc")
    }

    coarse_paths = {
        "UV": resource_filename("DoWnGAN", "./data/interim_2000-10-01_to_2013-09-30.nc")
    }

    logged_model = f'/home/nannau/msc/DoWnGAN/mlflow_experiments/0/{hash_code}/artifacts/Generator/Generator_{epoch_number}/'
    generated_ds_path = f'/home/nannau/msc/DoWnGAN/mlflow_experiments/0/{hash_code}/artifacts/generated_ds.nc'

    coarse_sf_path = resource_filename("DoWnGAN", "data/interim_roughness_2000-10-01_to_2013-09-30.nc")
    coarse_mask_path = resource_filename("DoWnGAN", "data/_grib2netcdf-webmars-public-svc-blue-005-6fe5cac1a363ec1525f54343b6cc9fd8-J3TMVF.nc")
    coarse_sp_path = resource_filename("DoWnGAN", "data/interim_surface_pressure_2000-10-01_to_2013-09-30.nc")
    coarse_cape_path = resource_filename("DoWnGAN", "data/regrid_era5_era_cape_2000-10-01-2013-09-30.nc")
    coarse_geo_path = resource_filename("DoWnGAN", "data/geopotential_era_interim.nc")

    region_area = {
        "florida": (4, 20, 70, 86),
        "central": (30, 46, 50, 66),
        "west": (30, 46, 15, 31)
    }



    def load_coarse(paths):
        # Load ERA Interim
        coarse = xr.open_dataset(coarse_paths["UV"], engine="scipy").astype("float")
        # Organize lats in increasing order:
        coarse = coarse.sortby("latitude", ascending=True)

        return coarse
        
    def open_covariate(path):
        return xr.open_dataset(path).sortby('latitude', ascending=True)

    def extend_invariant(field, ref):
        l = [field for i in range(ref.time.shape[0])]
        ds = xr.concat(l, dim="time")
        ds = ds.assign_coords(
                {
                "time": ref.time, 
                "latitude": ref.latitude, 
                "longitude": ref.longitude
                }
            ).transpose("time", "latitude", "longitude")

        return ds

    def mask_and_standardize_coarse(time_example, lat_example, lon_example, coarse_u10, coarse_v10, sf):


        times = dt_index(time_example)

        # Apply filter to times for months you'd like
        time_mask = ~filter_times(times, mask_years=[2000, 2006, 2010])

        # WRF has bad first field for some reason
        time_mask[0] = False


        coarse_sf = open_covariate(coarse_sf_path)
        coarse_sp = open_covariate(coarse_sp_path)

        coarse_cape = xr.open_dataset(coarse_cape_path)
        coarse_cape = coarse_cape.rename({"lat": "latitude", "lon": "longitude"})
        coarse_cape = coarse_cape.sortby("latitude", ascending=True).assign_coords({
            "time": coarse_u10.time, "latitude": coarse_u10.latitude, "longitude": coarse_u10.longitude
            })

        coarse_geo = open_covariate(coarse_geo_path)
        coarse_mask = open_covariate(coarse_mask_path)

        mlat1 = np.argmin(np.abs(lat_example.min()-coarse_mask.latitude).values)
        mlat2 = np.argmin(np.abs(lat_example.max()-coarse_mask.latitude).values)
        mlon1 = np.argmin(np.abs(lon_example.min()-(-360+coarse_mask.longitude)).values)
        mlon2 = np.argmin(np.abs(lon_example.max()-(-360+coarse_mask.longitude)).values)+1

        coarse_mask = coarse_mask.lsm[0, mlat1:mlat2, mlon1:mlon2]
        coarse_mask = extend_invariant(coarse_mask, coarse_u10)

        low, up, l, r = region_area[region]

        coarse_geo = extend_invariant(coarse_geo, coarse_u10)
        fields = [
            coarse_u10[time_mask, low:up, l:r],
            coarse_v10[time_mask, low:up, l:r],
            coarse_sp.sp[time_mask, low:up, l:r],
            coarse_sf.sr[time_mask, low:up, l:r],
            coarse_geo.z[time_mask, low:up, l:r],
            coarse_cape.cape[time_mask, low:up, l:r],
        ]

        fields = [xr_standardize_field(x) for x in fields]
        fields.insert(2, coarse_mask[time_mask, low:up, l:r])

        coarse = xr.concat(fields, dim="var").transpose('time', 'var', 'latitude', 'longitude')
        # coarse = xr.concat([coarse_u10_patch, coarse_v10_patch, coarse_mask, coarse_sp, coarse_sf], dim="var").transpose('time', 'var', 'latitude', 'longitude')
        # coarse = xr.concat([coarse_u10_patch, coarse_v10_patch], dim="var").transpose('time', 'var', 'latitude', 'longitude')


        return time_mask, coarse


    def gen_chunks(ds, fine_arr, coarse):

        ds = xr.concat([fine_arr, fine_arr], dim="var", coords="all").transpose("Times", "var", "lat", "lon")

        logging.info("Generating fake fields...")
        G = mlflow.pytorch.load_model(logged_model).to(device)
        state_dict = mlflow.pytorch.load_state_dict(logged_model)
        G.load_state_dict(state_dict)
        i1 = 0
        for chunk in torch.chunk(coarse, chunks=100):
            i2 = i1 + chunk.size(0)
            ds[i1:i2, ...] = G(chunk.to(device).float()).detach().cpu()
            i1 = i2
    
        logging.info("Writing chunks to file...")
        ds.chunk({"Times":5}).to_netcdf(generated_ds_path)

    def generate():
        sf = 8
        low, up, l, r = region_area[region]

        path_fine_u = glob.glob(fine_paths["U"])
        dsf = xr.open_mfdataset(
            path_fine_u,
            combine="by_coords",
            engine="h5netcdf",
            concat_dim="Time",
        )

        coarse = load_coarse(coarse_paths["UV"])
        coarse_u10 = coarse.u10.loc["2000-01-01":"2015-05-30"]
        coarse_v10 = coarse.v10.loc["2000-01-01":"2015-05-30"]
        time_mask, coarse = mask_and_standardize_coarse(dsf.Times, dsf.lat, dsf.lon, coarse_u10, coarse_v10, sf)
        coarse_t = torch.from_numpy(np.array(coarse))
        fine_arr = dsf.U10[time_mask, sf * low : sf * up, sf * l : sf * r]
        gen_chunks(dsf, fine_arr, coarse_t)


    generate()
    logging.info("Completed!")

if __name__ == "__main__":
    main()