import sys
import os
import dask
import tqdm
import argparse
from functools import partial
import numpy as np
import xarray as xr

try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")

vars = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "cat_non"]
path = ["/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_regression_v3_full_0.nc"]
location_hrrr = "/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp"
means_file = os.path.join(location_hrrr, 'stats', 'means.npy')
stds_file = os.path.join(location_hrrr, 'stats', 'stds.npy')


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

@dask.delayed
def process(i, forecast, truth, pred, n_ensemble):
    # Slice and load only the necessary data to avoid memory issues
    truth_slice = truth.isel(time=slice(i, i + 1), forecast=slice(forecast, forecast + 1)).load()
    pred_slice = pred.isel(time=slice(i, i + 1), forecast=slice(forecast, forecast + 1)).load()
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
    truth_rescaled = type_rescale_truth(truth_slice)
    pred_rescaled = type_rescale_pred(pred_slice)
    d = xskillscore.brier_score(truth_rescaled[vars[4:]], pred_rescaled[vars[4:]], member_dim="ensemble", dim=dim)

    # Concatenate metrics
    metrics = xr.concat([a, b, c, crps_mean, d], dim="metric") # Add forecast dimension

    return metrics

def main(path: str, output: str, n_ensemble: int = -1):
    for j, p in enumerate(path):
        truth, pred, root = open_samples(p)
        print(pred["time"], flush=True)

        tasks = []
        for i in tqdm.tqdm(range(truth.sizes["time"]), total=truth.sizes["time"]):
            forecast_metrics = []
            for forecast in range(0, 9):
                forecast_metrics.append(process(i, forecast, truth, pred, n_ensemble))
            
            # Concatenate along the forecast dimension
            time_metrics = dask.delayed(xr.concat)(forecast_metrics, dim="forecast")
            tasks.append(time_metrics)
        
        # Compute all tasks in parallel
        metrics = dask.compute(*tasks)
        metrics = xr.concat(metrics, dim="time")  # Concatenate along time dimension
        metrics.attrs["n_ensemble"] = n_ensemble
        print(metrics)

        # Save to NetCDF with a single-threaded scheduler
        with dask.config.set(scheduler="single-threaded"):
            metrics.to_netcdf(f"{output}_{j}", mode="w")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--n-ensemble", type=int, default=-1)
    args = parser.parse_args()

    main(path, args.output, args.n_ensemble)