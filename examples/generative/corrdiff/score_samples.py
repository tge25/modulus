# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score the generated samples

Saves a netCDF of CRPS and other scores. Depends on time, but space and ensemble have been reduced::

    netcdf scores {
dimensions:
        metric = 4 ;
        time = 205 ;
variables:
        double eastward_wind_10m(metric, time) ;
                eastward_wind_10m:_FillValue = NaN ;
        double maximum_radar_reflectivity(metric, time) ;
                maximum_radar_reflectivity:_FillValue = NaN ;
        double northward_wind_10m(metric, time) ;
                northward_wind_10m:_FillValue = NaN ;
        double temperature_2m(metric, time) ;
                temperature_2m:_FillValue = NaN ;
        int64 time(time) ;
                time:units = "hours since 1990-01-01" ;
                time:calendar = "standard" ;
        string metric(metric) ;
}
"""

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
vars = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain"]
path = ["/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_full_pmean_m1p2_new1_0.nc", "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_full_pmean_m1p2_new2_0.nc"]
location_hrrr = "/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp"
means_file = os.path.join(location_hrrr, 'stats', 'means.npy')
stds_file = os.path.join(location_hrrr, 'stats', 'stds.npy')

means_hrrr = np.load(means_file)[:8]
stds_hrrr = np.load(stds_file)[:8]

def open_samples(f):
    """
    Open prediction and truth samples from a dataset file and normalize them.

    Parameters:
        f: Path to the dataset file.
        means_hrrr: Array of mean values for normalization.
        stds_hrrr: Array of standard deviation values for normalization.

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

# compute metrics sequentially
def process(i, path, n_ensemble):
    truth, pred, root = open_samples(path)
    
    truth = truth.isel(time=slice(i,  i + 1)).load()
    pred = pred.isel(time=slice(i, i + 1)).load()
    dim = ["x", "y"]

    a = xskillscore.rmse(truth, pred.mean("ensemble"), dim=dim)
    b = xskillscore.crps_ensemble(truth, pred, member_dim="ensemble", dim=dim)

    c = pred.std("ensemble").mean(dim)
    crps_mean = xskillscore.crps_ensemble(
        truth,
        pred.mean("ensemble").expand_dims("ensemble"),
        member_dim="ensemble",
        dim=dim,
    )

    metrics = xr.concat([a, b, c, crps_mean], dim="metric").assign_coords(
        metric=["rmse", "crps", "std_dev", "mae"]
    ).load()

    return metrics

def main(path: str, output: str, n_ensemble: int = -1):

    for j, p in enumerate(path):
        truth, pred, root = open_samples(p)
        print(truth, flush=True)
        print(pred["time"], flush=True)

        metrics = {}
        for i in range(0,9):
            metrics[i] = []

        for i in tqdm.tqdm(range(truth.sizes["time"]), total=truth.sizes["time"]):
            metric = process(i, path[0], n_ensemble)
            metrics[i%9].append(metric)
            sys.stderr.flush()        
        print(metrics, flush=True)
        for i in range(0,9):
            metrics[i] = xr.concat(metrics[i], dim="time")
            metrics[i].attrs["n_ensemble"] = n_ensemble

            # to netcdf with single threaded scheduler to avoid deadlocks
            with dask.config.set(scheduler="single-threaded"):
                metrics[i].to_netcdf(f"{output}{i:02d}_{j}", mode="w")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("path", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--n-ensemble", type=int, default=-1)
    args = parser.parse_args()

    main(path, args.output, args.n_ensemble)
