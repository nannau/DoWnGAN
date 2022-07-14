# Time masking
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

def datetime_wrf_period(start_time, end_time):
    """
    Returns the WRF time period increasing by 6 hours
    """

    diff = end_time-start_time
    hours = int((diff.days * 24 + diff.seconds // 3600)/6)
    times = [start_time + timedelta(hours=i*6) for i in range(hours)]
    return times

def wrf_to_dt(times: list) -> np.ndarray:
    """
    Converts the WRF time to datetime.
    """
    # Only works for WRF style time data
    tlist = []
    for t in times:
        year = int(str(float(t))[:4])
        month = int(str(float(t))[4:6])
        day = int(str(float(t))[6:8])
        hours = int(np.round(24 * float(str(float(t))[8:])))
        tlist.append(np.datetime64(datetime(year, month, day, hours)))

    tlist = np.array(tlist).astype("datetime64[D]")
    tlist_pd = pd.DatetimeIndex(tlist)
    return np.array(tlist_pd)


def filter_times(times: list, mask_years=None) -> np.ndarray:
    """Returns boolean mask if years are in time index
    """
    times = np.array(times).astype("datetime64[D]")
    times = pd.DatetimeIndex(times)
    if mask_years is not None:
        print(f"Masking these years: {mask_years}")
        filter_func = np.vectorize(lambda x: True if x.year not in mask_years else False)
        times = filter_func(times.astype(object))

    return xr.DataArray(times, dims="time")

