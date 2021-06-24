import numpy as np


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
