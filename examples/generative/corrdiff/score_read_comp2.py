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
#ds_handle = xr.open_zarr(paths[name], consolidated=True
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

vars = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain"]
vars_baseline = ["10u", "10v", "2t"]
metrics_1 = np.zeros((5, 8, 9))
metrics_2 = np.zeros((5, 8, 9))
baseline_path = "/lustre/fsw/coreai_climate_earth2/eos-corrdiff-dir/scores/v1/HRRR_large_validation/patched_500kreg_partial/25004/scores.nc"

file_name_1 = f"/lustre/fsw/coreai_climate_earth2/corrdiff/scores/450samples_regression_v3_0"

file_name_3 = f"/lustre/fsw/coreai_climate_earth2/corrdiff/scores/450samples_regression_v4_0"

metric1 = xr.open_dataset(file_name_1)

metric3 = xr.open_dataset(file_name_3)

# Concatenate along time dimension
metric_1 = metric1
metric_2 = metric3

print(metric_1)
print(metric_2)

for i,var in enumerate(vars):
    metrics_1[:,i,] =  np.mean(metric_1[var].values, axis=1)
    metrics_2[:,i,] =  np.mean(metric_2[var].values, axis=1)

# Define the line styles and colors
line_styles = ['-', '--']  # Solid line for metrics_1, dashed line for metrics_2
colors = ['b', 'g', 'r', 'c']  # You can choose any color combination

print(metrics_1)
plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
for i, var in enumerate(vars[:4]):
    plt.plot(metrics_1[0][i], line_styles[0], color=colors[i], label=f'{var} v3')
    plt.plot(metrics_2[0][i], line_styles[1], color=colors[i], label=f'{var} v4')
plt.legend(loc='best')
plt.title("RMSE")

plt.subplot(2,3,2)
for i, var in enumerate(vars[:4]):
    plt.plot(metrics_1[1][i], line_styles[0], color=colors[i], label=f'{var} v3')
    plt.plot(metrics_2[1][i], line_styles[1], color=colors[i], label=f'{var} v4')
plt.legend(loc='best')
plt.title("CRPS")

plt.subplot(2,3,3)
for i, var in enumerate(vars[:4]):
    plt.plot(metrics_1[2][i], line_styles[0], color=colors[i], label=f'{var} v3')
    plt.plot(metrics_2[2][i], line_styles[1], color=colors[i], label=f'{var} v4')
plt.legend(loc='best')
plt.title("STD")

plt.subplot(2,3,4)
for i, var in enumerate(vars[:4]):
    plt.plot(metrics_1[3][i], line_styles[0], color=colors[i], label=f'{var} v3')
    plt.plot(metrics_2[3][i], line_styles[1], color=colors[i], label=f'{var} v4')
plt.legend(loc='best')
plt.title("MAE")

plt.subplot(2,3,5)
for i, var in enumerate(vars[4:]):
    plt.plot(metrics_1[4][i+4], line_styles[0], color=colors[i], label=f'{var} v3')
    plt.plot(metrics_2[4][i+4], line_styles[1], color=colors[i], label=f'{var} v4')
plt.legend(loc='best')
plt.title("Brier score")

plt.savefig("twc_mvp_regression_v3v4_validation.png")




