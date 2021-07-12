import torch
from torch.utils.data import Dataset
import xarray as xr


class NetCDFSR(Dataset):
    """Data loader for netCDF data for GAN Super Resolution"""

    def __init__(
        self,
        fine: xr.DataArray,
        coarse: xr.DataArray,
        Z: xr.DataArray,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            coarse (xarray.DataArray): Coarse resolution fields (time series)
            fine (xarray.DataArray): Fine resolution xarray.DataArray
            pcas (numpy.ndarray): Principal components
            device (torch.device): cuda device
        """
        self.fine = fine
        self.coarse = coarse
        self.Z = Z

    def __len__(self):
        return self.fine.size(0)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        fine_ = self.fine[idx, ...]
        coarse_ = self.coarse[idx, ...]

        Z_ = self.Z[idx, ...]

        return fine_, coarse_, Z_


def xr_standardize_field(field: xr.DataArray) -> xr.DataArray:
    """Standardize/regularize field assuming
    single 'colour' channel
    field (xarray.DataArray)
    """
    mean = field.mean(skipna=True)
    std = field.std(skipna=True)
    field = (field - mean) / std
    return field
