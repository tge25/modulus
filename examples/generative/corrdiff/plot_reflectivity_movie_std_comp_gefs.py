# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import math
import pickle
import xskillscore
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pylab as plt
import xarray
from matplotlib.colors import TwoSlopeNorm
import time
'''
time = ["2024011212f15", "2024011400f00", "2024011400f21", "2024030518f03",
        "2024051506f24", "2024061212f12", "2024070400f00", "2024070400f18",
        "2024070800f00", "2024070800f12", "2024071006f00", "2024030518f03",]
'''

time1 = ["2024012100f00","2024012100f03","2024012100f06","2024012100f09",
         "2024012100f12","2024012100f15","2024012100f18","2024012100f21",
         "2024012100f24","2024012200f00","2024012200f03","2024012200f06",
         "2024012200f09","2024012200f12","2024012200f15",
         "2024012200f18","2024012200f21","2024012200f24","2024012300f00",
         "2024012300f03","2024012300f06","2024012300f09","2024012300f12",
         "2024012300f15","2024012300f18","2024012300f21","2024012300f24",
         "2024012400f00","2024012400f03","2024012400f06","2024012400f09",
         "2024012400f12","2024012400f15","2024012400f18","2024012400f21",
         "2024012400f24","2024012500f00","2024012500f03","2024012500f06",
         "2024012500f09","2024012500f12","2024012500f15","2024012500f18",
         "2024012500f21","2024012500f24"]

time2 = ["2024040300f00","2024040300f03","2024040300f06","2024040300f09",
         "2024040300f12","2024040300f15","2024040300f18","2024040300f21",
         "2024040300f24","2024040400f00","2024040400f03","2024040400f06",
         "2024040400f09","2024040400f12","2024040400f15","2024040400f18",
         "2024040400f21","2024040400f24"]

times = time1 + time2

gefs_surface_channels = ["u10m", "v10m", "t2m", "q2m", "sp", "msl", "precipitable_water"]
gefs_isobaric_channels = ['u1000', 'u925', 'u850', 'u700', 'u500', 'u250', 'v1000', 'v925', 'v850', 'v700', 'v500', 'v250', 'z1000', 'z925', 'z850', 'z700', 'z500', 'z250', 't1000', 't925', 't850', 't700', 't500', 't250',  'q1000', 'q925', 'q850', 'q700', 'q500', 'q250']
hrrr_stats_channels = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "refc"]

path = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_diffusion_v3_winter_storm1_0.nc" #"image_outdir_val_paper_plot_0.nc"
output_name = "twc_mvp1_v3_schurn_p5"

ds = xarray.open_dataset(path)
lat = np.array(ds.variables["lat"])
lon = np.array(ds.variables["lon"])
print(lon)
ds_prediction = xarray.open_dataset(path, group="prediction")
ds_truth = xarray.open_dataset(path, group="truth")
ds_input = xarray.open_dataset(path, group="input")
print(ds_prediction)
ds_prediction = ds_prediction.assign_coords(
    time=ds["time"], lat=ds["lat"], lon=ds["lon"]
)
ds_truth = ds_truth.assign_coords(time=ds["time"], lat=ds["lat"], lon=ds["lon"])
ds_input = ds_input.assign_coords(time=ds["time"], lat=ds["lat"], lon=ds["lon"])

ds_hrrr = xarray.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp_valid/HRRR_forecasts_2024.zarr", consolidated=True)
ds_gefs = xarray.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp_ens_valid/GEFS_surface_2024_winter_storm.zarr", consolidated=True)

ds_gefs['x'] = ds_gefs.lat.values[:,0]
ds_gefs['y'] = ds_gefs.lon.values[0,:]%360
lat = ds_hrrr.lat
lon = ds_hrrr.lon%360
ds_gefs = ds_gefs.interp(x=lat, y=lon)
print(lon.values)
ds_hrrr = ds_hrrr.assign_coords(time=ds["time"], lat=ds_hrrr["lat"], lon=ds_hrrr["lon"])
ds_gefs = ds_gefs.assign_coords(time=ds["time"], lat=ds_hrrr["lat"], lon=ds_hrrr["lon"])
print(ds_gefs)

dim = ["x", "y"]
plt.rcParams.update({'font.size': 22})
nvars = 48 + 1
ncolumns = 4
nrows = 6
sequential_cmap = plt.get_cmap("magma", 20)

tic = time.time()
for movie in range(0,2):
    for i in range(0, 6):
        plt.figure(figsize=(42, 30))
        plt.style.use('dark_background')

        var = "precip"
        ax = plt.subplot(nrows, ncolumns, 1, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        im1 = ax.pcolormesh(
            lon,
            lat,
            np.log(ds_prediction[var].clip(min=0).mean(dim="ensemble")[i, :, :]),
            cmap=sequential_cmap,
            vmin=-3, vmax=3,
        )
        crps_mean = xskillscore.crps_ensemble(
            ds_truth[var][i, :, :], ds_prediction[var].mean(dim="ensemble")[i, :, :].expand_dims("ensemble"), member_dim="ensemble", dim=dim
        )
        plt.title("mean | MAE: %f"%crps_mean)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        ax.set_ylabel("latitude")
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False

        ax = plt.subplot(nrows, ncolumns, 2, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon,
            lat,
            np.sqrt(
                np.log(ds_prediction[var].clip(min=0).std(dim="ensemble")[i, :, :].clip(min=0))
            ),
            vmin=-3, vmax=2,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[i, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 3, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, np.log(ds_prediction[var][0, i, :, :].clip(min=0)), cmap=sequential_cmap,vmin=-3, vmax=3,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][i, :, :], ds_prediction[var][:, i, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 4, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, np.log(ds_truth[var][i, :, :].clip(min=0)), cmap=sequential_cmap, vmin=-3, vmax=3,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("target")
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "u10m"
        ax = plt.subplot(nrows, ncolumns, 5, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        im1 = ax.pcolormesh(
            lon,
            lat,
            ds_prediction[var].mean(dim="ensemble")[i, :, :],
            cmap=sequential_cmap,
            vmin=-20, vmax=20,
        )
        crps_mean = xskillscore.crps_ensemble(
            ds_truth[var][i, :, :], ds_prediction[var].mean(dim="ensemble")[i, :, :].expand_dims("ensemble"), member_dim="ensemble", dim=dim
        )
        plt.title("mean | MAE: %f"%crps_mean)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        ax.set_ylabel("latitude")
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False

        ax = plt.subplot(nrows, ncolumns, 6, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon,
            lat,
            np.sqrt(
                ds_prediction[var].std(dim="ensemble")[i, :, :].clip(min=0)
            ),
            vmin=0, vmax=5,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[i, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 7, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_prediction[var][0, i, :, :], cmap=sequential_cmap,vmin=-20, vmax=20,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][i, :, :], ds_prediction[var][:, i, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 8, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_truth[var][i, :, :], cmap=sequential_cmap,vmin=-20, vmax=20,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("target")
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "cat_rain"
        ax = plt.subplot(nrows, ncolumns, 9, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        im1 = ax.pcolormesh(
            lon,
            lat,
            ds_prediction[var].clip(min=0).mean(dim="ensemble")[i, :, :],
            cmap=sequential_cmap,
            vmin=0, vmax=1,
        )
        crps_mean = xskillscore.crps_ensemble(
            ds_truth[var][i, :, :], ds_prediction[var].mean(dim="ensemble")[i, :, :].expand_dims("ensemble"), member_dim="ensemble", dim=dim
        )
        plt.title("mean | MAE: %f"%crps_mean)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        ax.set_ylabel("latitude")
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False

        ax = plt.subplot(nrows, ncolumns, 10, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon,
            lat,
            np.sqrt(
                ds_prediction[var].clip(min=0).std(dim="ensemble")[i, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[i, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 11, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_prediction[var][0, i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][i, :, :], ds_prediction[var][:, i, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 12, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_truth[var][i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("target")
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False


        var = "cat_snow"
        ax = plt.subplot(nrows, ncolumns, 13, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        im1 = ax.pcolormesh(
            lon,
            lat,
            ds_prediction[var].clip(min=0).mean(dim="ensemble")[i, :, :],
            cmap=sequential_cmap,
            vmin=0, vmax=1,
        )
        crps_mean = xskillscore.crps_ensemble(
            ds_truth[var][i, :, :], ds_prediction[var].mean(dim="ensemble")[i, :, :].expand_dims("ensemble"), member_dim="ensemble", dim=dim
        )
        plt.title("mean | MAE: %f"%crps_mean)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        ax.set_ylabel("latitude")
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False

        ax = plt.subplot(nrows, ncolumns, 14, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon,
            lat,
            np.sqrt(
                ds_prediction[var].clip(min=0).std(dim="ensemble")[i, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[i, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 15, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_prediction[var][0, i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][i, :, :], ds_prediction[var][:, i, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 16, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_truth[var][i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.title("target")
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "cat_freez"
        ax = plt.subplot(nrows, ncolumns, 17, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        im1 = ax.pcolormesh(
            lon,
            lat,
            ds_prediction[var].clip(min=0).mean(dim="ensemble")[i, :, :],
            cmap=sequential_cmap,
            vmin=0, vmax=1,
        )
        crps_mean = xskillscore.crps_ensemble(
            ds_truth[var][i, :, :], ds_prediction[var].mean(dim="ensemble")[i, :, :].expand_dims("ensemble"), member_dim="ensemble", dim=dim
        )
        plt.title("mean | MAE: %f"%crps_mean)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        ax.set_ylabel("latitude")
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False

        ax = plt.subplot(nrows, ncolumns, 18, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon,
            lat,
            np.sqrt(
                ds_prediction[var].clip(min=0).std(dim="ensemble")[i, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[i, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 19, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_prediction[var][0, i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][i, :, :], ds_prediction[var][:, i, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 20, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_truth[var][i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.title("target")
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "cat_ice"
        ax = plt.subplot(nrows, ncolumns, 21, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        im1 = ax.pcolormesh(
            lon,
            lat,
            ds_prediction[var].clip(min=0).mean(dim="ensemble")[i, :, :],
            cmap=sequential_cmap,
            vmin=0, vmax=1,
        )
        crps_mean = xskillscore.crps_ensemble(
            ds_truth[var][i, :, :], ds_prediction[var].mean(dim="ensemble")[i, :, :].expand_dims("ensemble"), member_dim="ensemble", dim=dim
        )
        plt.title("mean | MAE: %f"%crps_mean)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        ax.set_ylabel("latitude")
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False

        ax = plt.subplot(nrows, ncolumns, 22, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon,
            lat,
            np.sqrt(
                ds_prediction[var].clip(min=0).std(dim="ensemble")[i, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[i, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 23, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_prediction[var][0, i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][i, :, :], ds_prediction[var][:, i, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 24, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        im1 = ax.pcolormesh(
            lon, lat, ds_truth[var][i, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.title("target")
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        plt.suptitle(f"{times[i][:4]}-{times[i][4:6]}-{times[i][6:8]} {times[i][8:10]}:00 | lead time: {times[i][11:]} hours", fontsize=40)
        plt.tight_layout()
        plt.savefig("./output_movie/reflectivity_movie%d.png"%i)
        plt.close()
        print(i, "save done", time.time()-tic, flush=True)

    import imageio 
    import os
    writer = imageio.get_writer(f'{output_name}_{movie}.mp4', fps = 3)
    dirFiles = os.listdir('./output_movie/') #list of directory files
    dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for im in dirFiles:
        writer.append_data(imageio.imread('./output_movie/'+im))
    writer.close()
    for im in dirFiles:
        os.remove('./output_movie/'+im)