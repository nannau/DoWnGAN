import torch
from torch.utils.data import Dataset
import xarray as xr


class NetCDFSR(Dataset):
    """Data loader for netCDF data for GAN Super Resolution"""

    def __init__(
        self,
        fine: xr.DataArray,
        coarse: xr.DataArray,
        pcas: xr.DataArray,
        pca_og_shape: xr.DataArray,
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
        self.pcas = pcas
        self.pca_og_shape = pca_og_shape
        self.Z = Z

    def __len__(self):
        return self.fine.size(0)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        arr = self.fine[idx, ...]
        carr = self.coarse[idx, ...]
        pca_og_shape = self.pca_og_shape
        Z = self.Z[idx, ...]

        pcas_arr = torch.unsqueeze(self.pcas, 0)

        return arr, carr, pcas_arr, pca_og_shape, Z


def xr_standardize_field(field: xr.DataArray) -> xr.DataArray:
    """Standardize/regularize field assuming
    single 'colour' channel
    field (xarray.DataArray)
    """
    mean = field.mean(skipna=True)
    std = field.std(skipna=True)
    field = (field - mean) / std
    return field
