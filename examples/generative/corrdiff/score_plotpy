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
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

vars = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain"]
vars_baseline = ["10u", "10v", "2t"]
metrics = np.zeros((4, 8, 9))
baseline_path = "/lustre/fsw/coreai_climate_earth2/eos-corrdiff-dir/scores/v1/HRRR_large_validation/patched_500kreg_partial/25004/scores.nc"

for time in range(0,9):
        file_name = f"scores_pmean_m1p2{time:02d}_0"
        metric1 = xr.open_dataset(file_name)
        file_name = f"scores_pmean_m1p2{time:02d}_1"
        metric2 = xr.open_dataset(file_name)
        metric = xr.concat((metric1, metric2), dim="time")

        for i,var in enumerate(vars):
                metrics[:,i,time] =  np.mean(metric[var].values, axis=1)

print(metrics)
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
for i,var in enumerate(vars):
        plt.plot(metrics[0][i])
plt.legend(vars)
plt.title("RMSE")

plt.subplot(2,2,2)
for i,var in enumerate(vars):
        plt.plot(metrics[1][i])
plt.legend(vars)
plt.title("CRPS")

plt.subplot(2,2,3)
for i,var in enumerate(vars):
        plt.plot(metrics[2][i])
plt.legend(vars)
plt.title("STD")

plt.subplot(2,2,4)
for i,var in enumerate(vars):
        plt.plot(metrics[3][i])
plt.legend(vars)
plt.title("MAE")

plt.savefig("twc_mvp_pmean_m1p2_validation1.png")


plt.figure(figsize=(14, 8))

# Adjusted to create 2x4 subplots
for i, var in enumerate(vars):
    data = []
    for time in range(0, 9):
        file_name_1 = f"scores_pmean_m1p2{time:02d}_0"
        file_name_2 = f"scores_pmean_m1p2{time:02d}_1"
        
        metric1 = xr.open_dataset(file_name_1)
        metric2 = xr.open_dataset(file_name_2)
        
        # Concatenate along time dimension
        metric = xr.concat((metric1, metric2), dim="time")
        
        # Collect data for the specific variable
        data.append(metric[var][1].values)
    if i<3:
        data.append(xr.open_dataset(baseline_path)[vars_baseline[i]][1].values)    
        ax = plt.subplot(2, 4, i + 1)
        ax.boxplot(data, labels=[0, 3, 6, 9, 12, 15, 18, 21, 24, "b"])
    else:
        ax = plt.subplot(2, 4, i + 1)
        ax.boxplot(data, labels=[0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_title(f"CRPS boxplot - {var}")

plt.tight_layout()
plt.savefig("crps_boxplot_pmean_m1p2.png")
plt.show()
