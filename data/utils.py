import os
import glob
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from osgeo import gdal
import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Bounds
xbounds = slice(-101, -88)
ybounds = slice(46.5, 39)

# Years
years = list(range(2010, 2024))


# Vars
ndvi_var_name = "1 km 16 days NDVI"
var_list = ["ndvi", "t2m", "tp"]
var_colors = {"ndvi": "forestgreen", "t2m": "maroon", "tp": "mediumblue"}
var_cmaps = {"ndvi": "RdYlGn", "t2m": "Reds", "tp": "Blues"}
var_titles = {"ndvi": "NDVI", "t2m": "Temperature", "tp": "Precipitation"}


def combine_resample_era5(
    filenames,
    resample_dict={"time": "8D"},
    xbounds=None,
    ybounds=None,
):
    """Combine & resample era5 data.

    Args:
        filenames (list): era5 paths.
        resample_dict (dict, optional): Dictionary of resampling option(s). Defaults to {"time": "8D"}.
        xbounds (slice, optional): Slice object to clip dataset longitudes by. Defaults to None.
        ybounds (slice, optional): Slice object to clip dataset latitudes by. Defaults to None.

    Returns:
        ds (xr.Dataset): era5 dataset.
    """
    ds = xr.merge([xr.open_dataset(file) for file in filenames])
    ds = ds.rename({"longitude": "x", "latitude": "y"})
    # Slice data by bounds (if applicable)
    if xbounds is not None:
        ds = ds.sel(x=xbounds)
    if ybounds is not None:
        ds = ds.sel(y=ybounds)
    ds = ds.resample(resample_dict).mean()
    return ds


def moving_average_3d(arr, n=30, axis=2):
    """
    Apply a moving average filter along a specified axis of a 3D array.

    Args:
        arr: 3D numpy array.
        n: Window size for the moving average.
        axis: Axis along which to apply the moving average.

    Returns:
        A 3D numpy array with the moving average applied along the specified axis.
    """
    return uniform_filter1d(arr, size=n, axis=axis, mode="reflect")


def combine_resample_ndvi(
    filenames,
    ndvi_var_name=ndvi_var_name,
    coarse_ds=None,
    resample_dict={"time": "8D"},
    xbounds=None,
    ybounds=None,
):
    """Combine NDVI files and coarsen, resample & clip as requested.

    Args:
        filenames (list): NDVI paths.
        ndvi_var_name (str, optional): NDVI variable name. Defaults to ndvi_var_name.
        coarse_ds (xr.Dataset, optional): Dataset to coarsen NDVI to, if desired. Defaults to None (no coarsening).
        resample_dict (dict, optional): Dictionary of resampling option(s). Defaults to {"time": "8D"}.
        xbounds (slice, optional): Slice object to clip dataset longitudes by. Defaults to None.
        ybounds (slice, optional): Slice object to clip dataset latitudes by. Defaults to None.

    Returns:
        ndvi_ds (xr.Dataset): NDVI dataset.
    """
    grid_locations = {file.split(".")[2] for file in filenames}
    dates = sorted(list({file.split(".")[1] for file in filenames}))
    print_val = f"Combining {len(filenames)} NDVI files ({len(grid_locations)} grids & {len(dates)} dates)"
    if coarse_ds is not None:
        print_val += (
            f" and coarsening to resolution: {np.diff(coarse_ds.x.values[:2])[0]}"
        )
    print(print_val)
    scale, nodataval = None, None
    ndvi_dss = []
    # Mosaic by date, then combine datasets on the time axis
    for date in tqdm(dates):
        files = [file for file in filenames if date in file]
        # Mosaic images together
        ndvi_ds = merge_arrays(
            [rxr.open_rasterio(file)[ndvi_var_name] for file in files]
        )
        # Reproject to EPSG:4326
        ndvi_ds = ndvi_ds.rio.reproject("EPSG:4326")
        # Slice data by bounds (if applicable)
        if xbounds is not None:
            ndvi_ds = ndvi_ds.sel(x=xbounds)
        if ybounds is not None:
            ndvi_ds = ndvi_ds.sel(y=ybounds)
        # To dataset & add time dimension
        ndvi_ds = ndvi_ds.to_dataset(name="ndvi").expand_dims("time")
        ndvi_ds["time"] = [
            datetime.strptime(
                rxr.open_rasterio(files[0]).attrs["RANGEBEGINNINGDATE"], "%Y-%m-%d"
            )
        ]
        # Set nodataval to nan for coarsening
        if nodataval is None:
            nodataval = ndvi_ds["ndvi"].rio.nodata
        ndvi_ds["ndvi"] = xr.DataArray(
            np.where(ndvi_ds["ndvi"] == nodataval, np.nan, ndvi_ds["ndvi"]),
            dims=ndvi_ds["ndvi"].dims,
        )
        # Coarsen to match coarse_ds spatial resolution -- if applicable
        if coarse_ds is not None:
            if scale is None:
                scale = int(
                    np.diff(coarse_ds.x.values[:2])[0]
                    / np.diff(ndvi_ds.x.values[:2])[0]
                )
            ndvi_ds = ndvi_ds.coarsen(x=scale, y=scale, boundary="trim").mean()
        ndvi_dss.append(ndvi_ds)
    ndvi_ds = xr.concat(ndvi_dss, dim="time").sortby("time")
    # Resample to an 8 day window, using a rolling mean
    ndvi_ds = ndvi_ds.resample(resample_dict).asfreq()
    # Rolling mean -- NOTE: this needs to be updated, currently will not work for non-default resampling options
    ndvi_ds = ndvi_ds.rolling(time=3, min_periods=1, center=True).mean()
    return ndvi_ds


def merge_era_and_ndvi(
    ndvi_ds, era_ds, var_list=var_list, plot_visual=True, path_out=None
):
    """Merge ERA and NDVI datasets. Option to plot visual of all variables.

    Args:
        ndvi_ds (xr.Dataset): NDVI dataset.
        era_ds (xr.Dataset): ERA datset.
        var_list (list, optional): List of variables (for plotting). Defaults to var_list.
        plot_visual (bool, optional): True to plot a map of the variables, else False. Defaults to True.
        path_out (str, optional): Path to output the visual to. Set to None to plot an inline figure. Defaults to None.

    Returns:
        ds (xr.Dataset): Combined dataset.
    """
    # Merge datasets (on x, y, time), matching by reindexing (nearest)
    ds = xr.merge(
        [
            ndvi_ds.isel(band=0),
            era_ds.reindex_like(ndvi_ds, method="nearest"),
        ]
    )
    # Visualized merge ds
    if plot_visual:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
        axes = axes.ravel()
        for i, var in enumerate(var_list):
            ds[var].isel(time=100).plot(cmap=var_cmaps[var], ax=axes[i])
            axes[i].set_title(var_titles[var])
        plt.tight_layout()
        plt.title("Study Area Variables")
        if path_out is None:

            plt.show()
        else:
            fig.savefig(path_out)
    return ds


def output_arrays(ds, var_list=var_list, folder_path_out="arrays"):
    """Output data arrays to .npy files. Outputs variables as well as dimensions x, y, & time.

    Args:
        ds (xr.Dataset): Dataset that contains variables.
        var_list (list, optional): List of variables to output. Defaults to global var_list.
        folder_path_out (str, optional): Folder to output arrays to. Defaults to "arrays".
    """
    os.makedirs(folder_path_out, exist_ok=True)
    # Output all variables
    for var in var_list:
        arr = ds[var].values
        # Bring ndvi into the -2 - 1 range
        if var == "ndvi":
            arr /= 10000
        path_out = os.path.join(folder_path_out, var)
        np.save(path_out, arr)
    # Output all dimensions
    for dim in ["x", "y", "time"]:
        np.save(os.path.join(folder_path_out, dim), ds[dim].values)


def corsen_dataset(ds, coarse_ds, method="first"):
    """Coarsen a dataset (dimensions: ["x", "y"]) to match another.

    Args:
        ds (xr.Dataset): Dataset to coarsen.
        coarse_ds (xr.Dataset): Dataset used to coarsen ds.
        method (str, optional): Method to sample by. One of either "mean" or "first". Defaults to "first".

    Returns:
        ds (xr.Dataset): Coarsened dataset.
    """
    # Map coarse ds coordinate pairs to high res ds coordinate pairs
    coarse_ds["y_c"] = coarse_ds["y"]
    coarse_ds["x_c"] = coarse_ds["x"]
    coarse_ds = coarse_ds.reindex_like(ds, method="nearest")
    ds["x"] = coarse_ds["x_c"]
    ds["y"] = coarse_ds["y_c"]
    # Create gridcell object & groupby over it (then unstack to go back to coordinate dimensions)
    ds = ds.stack(gridcell=["y", "x"])
    if method == "first":
        ds = ds.groupby("gridcell").first().unstack()
    elif method == "mean":
        ds = ds.groupby("gridcell").mean().unstack()
    return ds
