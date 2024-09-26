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

time1 = ["2024012000f24","2024012006f24","2024012012f24","2024012018f24",
         "2024012100f24","2024012106f24","2024012112f24","2024012118f24",
         "2024012200f24","2024012206f24","2024012212f24","2024012218f24",
         "2024012300f24","2024012306f24","2024012312f24","2024012318f24",
         "2024012400f24","2024012406f24","2024012412f24","2024012418f24"]

time2 = ["2024012100f00","2024012106f00","2024012112f00","2024012118f00",
         "2024012200f00","2024012206f00","2024012212f00","2024012218f00",
         "2024012300f00","2024012306f00","2024012312f00","2024012318f00",
         "2024012400f00","2024012406f00","2024012412f00","2024012418f00",
         "2024012500f00","2024012506f00","2024012512f00","2024012518f00"]

times = time1 + time2

gefs_surface_channels = ["u10m", "v10m", "t2m", "q2m", "sp", "msl", "precipitable_water"]
gefs_isobaric_channels = ['u1000', 'u925', 'u850', 'u700', 'u500', 'u250', 'v1000', 'v925', 'v850', 'v700', 'v500', 'v250', 'z1000', 'z925', 'z850', 'z700', 'z500', 'z250', 't1000', 't925', 't850', 't700', 't500', 't250',  'q1000', 'q925', 'q850', 'q700', 'q500', 'q250']
hrrr_stats_channels = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "refc"]

path = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_diffusion_v3_movie_leadtimecomp_0.nc" #"image_outdir_val_paper_plot_0.nc"
output_name = "twc_mvp1_v3_lead_time_comp_moive"

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
plt.rcParams.update({'font.size': 30})
nvars = 48 + 1
ncolumns = 6
nrows = 6
sequential_cmap = plt.get_cmap("magma", 20)

tic = time.time()
for i in range(len(time1)):
    plt.figure(figsize=(56, 30))
    plt.style.use('dark_background')
    
    for j in range(0,2):
        timestep = 2*i + 1 - j
        add = j*3
        var = "precip"
        ax = plt.subplot(nrows, ncolumns, 1+add, projection=ccrs.PlateCarree())
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
                np.log(ds_prediction[var].clip(min=0).std(dim="ensemble")[timestep, :, :].clip(min=0))
            ),
            vmin=-3, vmax=2,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[timestep, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 2+add, projection=ccrs.PlateCarree())
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
            lon, lat, np.log(ds_prediction[var][0, timestep, :, :].clip(min=0)), cmap=sequential_cmap,vmin=-3, vmax=3,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][timestep, :, :], ds_prediction[var][:, timestep, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 3+add, projection=ccrs.PlateCarree())
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
            lon, lat, np.log(ds_truth[var][timestep, :, :].clip(min=0)), cmap=sequential_cmap, vmin=-3, vmax=3,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("target")
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "u10m"
        ax = plt.subplot(nrows, ncolumns, 7+add, projection=ccrs.PlateCarree())
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
                ds_prediction[var].std(dim="ensemble")[timestep, :, :].clip(min=0)
            ),
            vmin=0, vmax=5,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[timestep, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 8+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_prediction[var][0, timestep, :, :], cmap=sequential_cmap,vmin=-20, vmax=20,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][timestep, :, :], ds_prediction[var][:, timestep, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 9+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_truth[var][timestep, :, :], cmap=sequential_cmap,vmin=-20, vmax=20,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("target")
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "cat_rain"
        ax = plt.subplot(nrows, ncolumns, 13+add, projection=ccrs.PlateCarree())
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
                ds_prediction[var].clip(min=0).std(dim="ensemble")[timestep, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[timestep, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 14+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_prediction[var][0, timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][timestep, :, :], ds_prediction[var][:, timestep, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 15+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_truth[var][timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("target")
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False


        var = "cat_snow"
        ax = plt.subplot(nrows, ncolumns, 19+add, projection=ccrs.PlateCarree())
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
                ds_prediction[var].clip(min=0).std(dim="ensemble")[timestep, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[timestep, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 20+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_prediction[var][0, timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][timestep, :, :], ds_prediction[var][:, timestep, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 21+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_truth[var][timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.title("target")
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "cat_freez"
        ax = plt.subplot(nrows, ncolumns, 25+add, projection=ccrs.PlateCarree())
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
                ds_prediction[var].clip(min=0).std(dim="ensemble")[timestep, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[timestep, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 26+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_prediction[var][0, timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][timestep, :, :], ds_prediction[var][:, timestep, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 27+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_truth[var][timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.title("target")
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        var = "cat_ice"
        ax = plt.subplot(nrows, ncolumns, 31+add, projection=ccrs.PlateCarree())
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
                ds_prediction[var].clip(min=0).std(dim="ensemble")[timestep, :, :].clip(min=0)
            ),
            vmin=0, vmax=1,
            cmap=sequential_cmap,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        c = ds_prediction[var].std(dim="ensemble")[timestep, :, :].mean(dim)
        plt.title("std | STD: %f"%c)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 32+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_prediction[var][0, timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.colorbar(im1, ax=ax, shrink=0.5)
        b = xskillscore.crps_ensemble(ds_truth[var][timestep, :, :], ds_prediction[var][:, timestep, :, :], member_dim="ensemble", dim=dim)
        plt.title("mem 0 | CRPS: %f"%b)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 33+add, projection=ccrs.PlateCarree())
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
            lon, lat, ds_truth[var][timestep, :, :].clip(min=0), cmap=sequential_cmap,vmin=0, vmax=1,
        )
        plt.title("target")
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

    plt.suptitle(f"{time2[i][:4]}-{time2[i][4:6]}-{time2[i][6:8]} {time2[i][8:10]}:00 | lead time: {time2[i][11:]} hours (reanalysis)                                                           {time1[i][:4]}-{time1[i][4:6]}-{time1[i][6:8]} {time1[i][8:10]}:00 | lead time: {time1[i][11:]} hours (forecast)", fontsize=45)
    plt.tight_layout()
    plt.savefig("./output_movie/reflectivity_movie%d.png"%i)
    plt.close()
    print(i, "save done", time.time()-tic, flush=True)

import imageio 
import os
writer = imageio.get_writer(f'{output_name}.mp4', fps = 3)
dirFiles = os.listdir('./output_movie/') #list of directory files
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for im in dirFiles:
    writer.append_data(imageio.imread('./output_movie/'+im))
writer.close()
for im in dirFiles:
    os.remove('./output_movie/'+im)