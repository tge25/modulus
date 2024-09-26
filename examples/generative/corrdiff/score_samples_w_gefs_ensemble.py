import sys
import os
import dask
import tqdm
import argparse
from functools import partial
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")

vars = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "cat_non"]
path = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_diffusion_v3_winter_storm1_0.nc"
location_hrrr = "/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp"
means_file = os.path.join(location_hrrr, 'stats', 'means.npy')
stds_file = os.path.join(location_hrrr, 'stats', 'stds.npy')

ds_hrrr = xr.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp_valid/HRRR_forecasts_2024.zarr", consolidated=True)
ds_gefs = xr.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp_ens_valid/GEFS_surface_2024_winter_storm.zarr", consolidated=True)
def type_rescale_pred(input):
    type_fields = [input[var].values for var in vars[4:]]
    type_fields = np.stack(type_fields, axis=0)
    type_fields[type_fields < 0] = 0
    type_fields[type_fields > 1] = 1
    type_fields = type_fields / np.max(type_fields, axis=0, keepdims=True)
    for i, var in enumerate(vars[4:]):
        input[var] = (('ensemble', 'time', 'forecast', 'y', 'x'), type_fields[i])
    return input

def type_rescale_truth(input):
    type_fields = [input[var].values for var in vars[4:]]
    type_fields = np.stack(type_fields, axis=0)
    type_fields = type_fields / np.max(type_fields, axis=0, keepdims=True)
    type_fields[type_fields < 0] = 0
    type_fields[type_fields > 1] = 1
    for i, var in enumerate(vars[4:]):
        input[var] = (('time', 'forecast', 'y', 'x'), type_fields[i])
    return input

def open_samples(f):
    """
    Open prediction and truth samples from a dataset file and normalize them.

    Parameters:
        f: Path to the dataset file.

    Returns:
        tuple: A tuple containing normalized truth, prediction, and root datasets.
    """
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])

    return truth, pred, root


def process(i, forecast, truth, pred, n_ensemble, is_pred = True):
    tic = time.time()
    # Slice and load only the necessary data to avoid memory issues
    truth_slice = truth.isel(time=slice(i,i+1), forecast=slice(forecast,forecast+1)).load()
    pred_slice = pred.isel(time=slice(i,i+1), forecast=slice(forecast,forecast+1)).load()
    if not is_pred:
        pred_slice['x'] = pred_slice.lat.values[:,0]
        pred_slice['y'] = pred_slice.lon.values[0,:]%360
        lat = truth_slice.lat
        lon = truth_slice.lon%360
        pred_slice = pred_slice.interp(x=lat, y=lon)
        truth_slice = truth_slice["values"]
        pred_slice = pred_slice["values"]  

    dim = ["x", "y"]
    # Calculate metrics
    a = xskillscore.rmse(truth_slice, pred_slice.mean("ensemble"), dim=dim)
    b = xskillscore.crps_ensemble(truth_slice, pred_slice, member_dim="ensemble", dim=dim)
    c = pred_slice.std("ensemble").mean(dim)
    crps_mean = xskillscore.crps_ensemble(
        truth_slice,
        pred_slice.mean("ensemble").expand_dims("ensemble"),
        member_dim="ensemble",
        dim=dim,
    )
    # Rescale prediction and truth
    if is_pred:
        truth_slice = type_rescale_truth(truth_slice)[vars[4:]]
        pred_slice = type_rescale_pred(pred_slice)[vars[4:]]
    else:
        truth_slice = truth_slice.isel(channel=slice(4, 8))
        pred_slice = pred_slice.isel(channel=slice(4, 8))
        
    d = xskillscore.brier_score(truth_slice, pred_slice, member_dim="ensemble", dim=dim)
    # Concatenate metrics
    metrics = xr.concat([a, b, c, crps_mean, d], dim="metric") # Add forecast dimension

    print(metrics)
    return metrics

def main():
    n_ensemble = -1

    if os.path.isfile("gefs"):
        metrics_gefs = xr.open_dataset("gefs")
        print(metrics_gefs.channel)
    else:
        metrics_gefs = []
        for i in tqdm.tqdm(range(6), total=6):
            forecast_metrics = []
            for forecast in range(0, 8):
                forecast_metrics.append(process(i, forecast, ds_hrrr, ds_gefs, n_ensemble, is_pred=False))
            # Concatenate along the forecast dimension
            forecast_metrics = xr.concat(forecast_metrics, dim="forecast")
            metrics_gefs.append(forecast_metrics)
        
        # Compute all tasks in parallel
        metrics_gefs = xr.concat(metrics_gefs, dim="time")  # Concatenate along time dimension
        metrics_gefs.attrs["n_ensemble"] = n_ensemble

        # Save to NetCDF with a single-threaded scheduler
        metrics_gefs.to_netcdf(f"gefs", mode="w")

    if os.path.isfile("pred_v3"):
        metrics_pred = xr.open_dataset("pred_v3")
        print(metrics_pred)
    else:
        truth, pred, root = open_samples(path)
        metrics_pred = []
        for i in tqdm.tqdm(range(6), total=truth.sizes["time"]):
            forecast_metrics = []
            for forecast in range(0, 9):
                forecast_metrics.append(process(i, forecast, truth, pred, n_ensemble))
            
            # Concatenate along the forecast dimension
            forecast_metrics = xr.concat(forecast_metrics, dim="forecast")
            metrics_pred.append(forecast_metrics)
        
        # Compute all tasks in parallel
        metrics_pred = xr.concat(metrics_pred, dim="time")  # Concatenate along time dimension
        metrics_pred.attrs["n_ensemble"] = n_ensemble
        # Save to NetCDF with a single-threaded scheduler
        with dask.config.set(scheduler="single-threaded"):
            metrics_pred.to_netcdf(f"pred_v3", mode="w")

    metrics_1 = np.full((5, 8, 9), np.nan)
    metrics_2 = np.full((5, 8, 9), np.nan)

    metrics_gefs_reorder = np.full((5, 6, 8, 9), np.nan)
    metrics_gefs_reorder[:,:,3,1:] = metrics_gefs["values"].values[:,:,:,4]
    metrics_gefs_reorder[:,:,2,1:] = metrics_gefs["values"].values[:,:,:,5]
    metrics_gefs_reorder[:,:,0,1:] = metrics_gefs["values"].values[:,:,:,6]
    metrics_gefs_reorder[:,:,1,1:] = metrics_gefs["values"].values[:,:,:,7]
    metrics_gefs_reorder[:,:,6,1:] = metrics_gefs["values"].values[:,:,:,0]
    metrics_gefs_reorder[:,:,5,1:] = metrics_gefs["values"].values[:,:,:,1]
    metrics_gefs_reorder[:,:,7,1:] = metrics_gefs["values"].values[:,:,:,2]
    metrics_gefs_reorder[:,:,4,1:] = metrics_gefs["values"].values[:,:,:,3]

    metrics_1 = np.mean(metrics_gefs_reorder, axis=1)
    for i,var in enumerate(vars[:-1]):
        metrics_2[:,i,] = np.mean(metrics_pred[var].values, axis=1)

    # Define the line styles and colors
    line_styles = ['--', '-']  # Solid line for metrics_1, dashed line for metrics_2
    colors = ['b', 'g', 'r', 'c']  # You can choose any color combination

    plt.figure(figsize=(15,12))
    plt.subplot(2,3,1)
    for i, var in enumerate(vars[:4]):
        plt.plot(metrics_1[0][i], line_styles[0], color=colors[i], label=f'{var} gefs')
        plt.plot(metrics_2[0][i], line_styles[1], color=colors[i], label=f'{var} v3')
    plt.legend(loc='best')
    plt.title("RMSE")

    plt.subplot(2,3,2)
    for i, var in enumerate(vars[:4]):
        plt.plot(metrics_1[1][i], line_styles[0], color=colors[i], label=f'{var} gefs')
        plt.plot(metrics_2[1][i], line_styles[1], color=colors[i], label=f'{var} v3')
    plt.legend(loc='best')
    plt.title("CRPS")

    plt.subplot(2,3,3)
    for i, var in enumerate(vars[:4]):
        plt.plot(metrics_1[2][i], line_styles[0], color=colors[i], label=f'{var} gefs')
        plt.plot(metrics_2[2][i], line_styles[1], color=colors[i], label=f'{var} v3')
    plt.legend(loc='best')
    plt.title("STD")

    plt.subplot(2,3,4)
    for i, var in enumerate(vars[:4]):
        plt.plot(metrics_1[3][i], line_styles[0], color=colors[i], label=f'{var} gefs')
        plt.plot(metrics_2[3][i], line_styles[1], color=colors[i], label=f'{var} v3')
    plt.legend(loc='best')
    plt.title("MAE")

    plt.subplot(2,3,5)
    for i, var in enumerate(vars[4:-1]):
        plt.plot(metrics_1[4][i+4], line_styles[0], color=colors[i], label=f'{var} gefs')
        plt.plot(metrics_2[4][i+4], line_styles[1], color=colors[i], label=f'{var} v3')
    plt.legend(loc='best')
    plt.title("Brier score")

    plt.savefig("twc_mvp_v3_winter_storm_validation.png")

    plt.figure(figsize=(22, 8))
    # Adjusted to create 2x4 subplots
    for i, var in enumerate(vars[:-1]):
        data = []
        for j in range(9):
            if i<4:
                data.append(metrics_gefs_reorder[1,:,i,j])
                data.append(metrics_pred[var][1,:,j].values)
            else:
                data.append(metrics_gefs_reorder[4,:,i,j])
                data.append(metrics_pred[var][4,:,j].values)

        # Create positions array with pairs closer together
        positions = []
        for j in range(len(data)//2):
            positions.append(1 + 2.2 * j)  # Position for v2
            positions.append(1 + 2.2 * j + 0.7)  # Position for v3
        
        ax = plt.subplot(2, 4, i + 1)
        
        ax.boxplot(data, positions=positions)
        
        # Adjust tick locations based on the number of data points and labels
        tick_locations = [1.25 + 2.2 * j for j in range(len(data)//2)]
        tick_labels = [f"{t}h" for t in range(0, 25, 3)]
        
        ax.set_xticks(tick_locations)  # Set the tick locations
        ax.set_xticklabels(tick_labels)  # Set the corresponding labels

        if i<4:
            ax.set_title(f"CRPS boxplot - {var}")
        else:
            ax.set_title(f"Brier score boxplot - {var}")

    plt.tight_layout()
    plt.savefig("crps_boxplot_v3_winter_storm.png")
    plt.show()

if __name__ == "__main__":
    main()