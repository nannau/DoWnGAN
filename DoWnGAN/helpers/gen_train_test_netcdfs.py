# Generates netcdf training and test files for GAN
from DoWnGAN.config import config

import DoWnGAN.helpers.gen_experiment_datasets as ged
import dask
from dask.distributed import Client, LocalCluster

import multiprocessing

# TODO:
# - Add netcdf generator. Identify how useful this actually is. 

def gen_netcdf():
    # Takes configuration from constants.py
    coarse_train, fine_train, coarse_test, fine_test = ged.generate_train_test_coarse_fine()

    # Write to netcdf
    print("Writing to netcdf...")
    print("Coarse train...")
    coarse_train.to_netcdf(config.PROC_DATA+f'/coarse_train_{config.region}.nc', engine="netcdf4")
    print("Fine train...")
    fine_train.chunk({"time": config.chunk_size, "lat": config.chunk_size, "lon":config.chunk_size}).to_netcdf(config.PROC_DATA+f'/fine_train_{config.region}.nc', engine="netcdf4")
    print("Coarse test...")
    coarse_test.to_netcdf(config.PROC_DATA+f'/coarse_test_{config.region}.nc', engine="netcdf4")
    print("Fine test...")
    fine_test.chunk({"time": config.chunk_size, "lat": config.chunk_size, "lon":config.chunk_size}).to_netcdf(config.PROC_DATA+f'/fine_test_{config.region}.nc', engine="netcdf4")

if __name__ == "__main__":
    # set up cluster and workers
    cores = int(multiprocessing.cpu_count()/2)
    print(f"Using {cores} cores")
    client = Client(n_workers = cores, threads_per_worker = 2, memory_limit='12GB')
    gen_netcdf()
        