# Generates netcdf training and test files for GAN
import gen_experiment_datasets as ged
import dask
from dask.distributed import Client, LocalCluster
import constants as c

import multiprocessing

# TODO:
# - Add netcdf generator. Identify how useful this actually is. 

def gen_netcdf():
    # Takes configuration from constants.py
    coarse_train, fine_train, coarse_test, fine_test = ged.load_original_netcdfs()

    # Write to netcdf
    print("Writing to netcdf...")
    print("Coarse train...")
    coarse_train.to_netcdf(c.proc_path+f'/coarse_train_{c.region}.nc', engine="netcdf4")
    print("Fine train...")
    fine_train.chunk({"time": c.chunk_size, "lat": c.chunk_size, "lon":c.chunk_size}).to_netcdf(c.proc_path+f'/fine_train_{c.region}.nc', engine="netcdf4")
    print("Coarse test...")
    coarse_test.to_netcdf(c.proc_path+f'/coarse_test_{c.region}.nc', engine="netcdf4")
    print("Fine test...")
    fine_test.chunk({"time": c.chunk_size, "lat": c.chunk_size, "lon":c.chunk_size}).to_netcdf(c.proc_path+f'/fine_test_{c.region}.nc', engine="netcdf4")

if __name__ == "__main__":
    # set up cluster and workers
    # dask.distributed.nanny["MALLOC_TRIM_THRESHOLD_"] = 0
    cores = int(multiprocessing.cpu_count()/2)
    print(f"Using {cores} cores")
    client = Client(n_workers = cores, threads_per_worker = 2, memory_limit='12GB')
    gen_netcdf()
        