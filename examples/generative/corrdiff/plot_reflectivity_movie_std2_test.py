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

import xskillscore
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import xarray
import time
import os
'''
time = ["2024011212f15", "2024011400f00", "2024011400f21", "2024030518f03",
        "2024051506f24", "2024061212f12", "2024070400f00", "2024070400f18",
        "2024070800f00", "2024070800f12", "2024071006f00", "2024030518f03",]
'''

time1 = ["2024012100","2024012200","2024012300", "2024012400", "2024012500"]

time2 = ["2024040300","2024040400"]

times = time1 + time2

gefs_surface_channels = ["u10m", "v10m", "t2m", "q2m", "sp", "msl", "precipitable_water"]
gefs_isobaric_channels = ['u1000', 'u925', 'u850', 'u700', 'u500', 'u250', 'v1000', 'v925', 'v850', 'v700', 'v500', 'v250', 'z1000', 'z925', 'z850', 'z700', 'z500', 'z250', 't1000', 't925', 't850', 't700', 't500', 't250',  'q1000', 'q925', 'q850', 'q700', 'q500', 'q250']
hrrr_stats_channels = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "refc"]

path = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_diffusion_v3_movie_winter_storms2_0.nc" #"image_outdir_val_paper_plot_0.nc"
output_name = "twc_mvp1_v3_schurn_p5"

ds = xarray.open_dataset(path)
lat = np.array(ds.variables["lat"])
lon = np.array(ds.variables["lon"])
ds_prediction = xarray.open_dataset(path, group="prediction")
ds_truth = xarray.open_dataset(path, group="truth")
ds_input = xarray.open_dataset(path, group="input")
ds_prediction = ds_prediction.assign_coords(
    time=ds["time"], lat=ds["lat"], lon=ds["lon"]
)
ds_truth = ds_truth.assign_coords(time=ds["time"], lat=ds["lat"], lon=ds["lon"])
ds_input = ds_input.assign_coords(time=ds["time"], lat=ds["lat"], lon=ds["lon"])

dim = ["x", "y"]
plt.rcParams.update({'font.size': 22})
nvars = 48 + 1
ncolumns = 2
nrows = 6
sequential_cmap = plt.get_cmap("magma", 20)
output_path = "corrdiff_output"
os.makedirs(output_path, exist_ok=True)

tic = time.time()
for event in range(0,2):
    if event == 0:
        start = 0
        end = len(time1)
    elif event == 1:
        start = len(time1)
        end = len(time1) + len(time2)
    
    for i in range(start, end):

        for forecast in range(9):
            plt.figure(figsize=(25, 30))
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
            if i == nvars - 1:
                ax.set_xlabel("longitude")
            else:
                gl.bottom_labels = False
            im1 = ax.pcolormesh(
                lon, lat, np.log(ds_prediction[var][0, i, forecast, :, :].clip(min=0)), cmap=sequential_cmap,vmin=-3, vmax=3,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            b = xskillscore.crps_ensemble(ds_truth[var][i, forecast, :, :], ds_prediction[var][:, i, forecast, :, :], member_dim="ensemble", dim=dim)
            plt.title("mem 0 | CRPS: %f"%b)
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

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
                lon, lat, np.log(ds_truth[var][i, forecast, :, :].clip(min=0)), cmap=sequential_cmap, vmin=-3, vmax=3,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            plt.title("target")
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

            var = "u10m"
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
                lon, lat, ds_prediction[var][0, i, forecast, :, :], cmap=sequential_cmap,vmin=-20, vmax=20,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            b = xskillscore.crps_ensemble(ds_truth[var][i, forecast, :, :], ds_prediction[var][:, i, forecast, :, :], member_dim="ensemble", dim=dim)
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
                lon, lat, ds_truth[var][i, forecast, :, :], cmap=sequential_cmap,vmin=-20, vmax=20,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            plt.title("target")
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

            var = "cat_rain"
            ax = plt.subplot(nrows, ncolumns, 5, projection=ccrs.PlateCarree())
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
                lon, lat, ds_prediction[var][0, i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            b = xskillscore.crps_ensemble(ds_truth[var][i, forecast, :, :], ds_prediction[var][:, i, forecast, :, :], member_dim="ensemble", dim=dim)
            plt.title("mem 0 | CRPS: %f"%b)
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

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
                lon, lat, ds_truth[var][i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            plt.title("target")
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False


            var = "cat_snow"
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
                lon, lat, ds_prediction[var][0, i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            b = xskillscore.crps_ensemble(ds_truth[var][i, forecast, :, :], ds_prediction[var][:, i, forecast, :, :], member_dim="ensemble", dim=dim)
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
                lon, lat, ds_truth[var][i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.title("target")
            plt.colorbar(im1, ax=ax, shrink=0.5)
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

            var = "cat_freez"
            ax = plt.subplot(nrows, ncolumns, 9, projection=ccrs.PlateCarree())
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
                lon, lat, ds_prediction[var][0, i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            b = xskillscore.crps_ensemble(ds_truth[var][i, forecast, :, :], ds_prediction[var][:, i, forecast, :, :], member_dim="ensemble", dim=dim)
            plt.title("mem 0 | CRPS: %f"%b)
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

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
                lon, lat, ds_truth[var][i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.title("target")
            plt.colorbar(im1, ax=ax, shrink=0.5)
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

            var = "cat_ice"
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
                lon, lat, ds_prediction[var][0, i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.colorbar(im1, ax=ax, shrink=0.5)
            b = xskillscore.crps_ensemble(ds_truth[var][i, forecast, :, :], ds_prediction[var][:, i, forecast, :, :], member_dim="ensemble", dim=dim)
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
                lon, lat, ds_truth[var][i, forecast, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
            )
            plt.title("target")
            plt.colorbar(im1, ax=ax, shrink=0.5)
            gl.right_labels = False
            gl.top_labels = False
            ax.coastlines(linewidth=0.5, color="white")
            gl.left_labels = False

            plt.suptitle(f"{times[i][:4]}-{times[i][4:6]}-{times[i][6:8]} {times[i][8:10]}:00 | lead time: {forecast*3:02d} hours", fontsize=40)
            plt.tight_layout()
            plt.savefig(f"./{output_path}/reflectivity_movie{i*9+forecast}.png")
            plt.close()
            print(i, "save done", time.time()-tic, flush=True)