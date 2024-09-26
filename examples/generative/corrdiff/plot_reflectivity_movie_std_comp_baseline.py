import datetime
import math
import pickle
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
import time as ttime
tic = ttime.time()
time = ["2024012100", "2024012118", "2024012212", "2024012306",
        "2024012400", "2024012418", ]

gefs_surface_channels = ["u10m", "v10m", "t2m", "q2m", "sp", "msl", "precipitable_water"]
gefs_isobaric_channels = ['u1000', 'u925', 'u850', 'u700', 'u500', 'u250', 'v1000', 'v925', 'v850', 'v700', 'v500', 'v250', 'z1000', 'z925', 'z850', 'z700', 'z500', 'z250', 't1000', 't925', 't850', 't700', 't500', 't250',  'q1000', 'q925', 'q850', 'q700', 'q500', 'q250']
hrrr_stats_channels = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "refc"]

path = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_diffusion_v3_winter_storm1_0.nc"
output_name = "twc_mvp1_comp_gefs_movie"

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

ds_hrrr = xarray.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp_valid/HRRR_forecasts_2024.zarr", consolidated=True)
ds_gefs = xarray.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp_ens_valid/GEFS_surface_2024_winter_storm.zarr", consolidated=True)
print(ds_prediction)
ds_gefs['x'] = ds_gefs.lat.values[:,0]
ds_gefs['y'] = ds_gefs.lon.values[0,:]%360
lat = ds_hrrr.lat
lon = ds_hrrr.lon%360
ds_gefs = ds_gefs.interp(x=lat, y=lon)
ds_gefs = ds_gefs["values"].values
ds_hrrr = ds_hrrr["values"].values
ds_gefs = ds_gefs[..., 1:1057, 4:1796]
ds_hrrr = ds_hrrr[..., 1:1057, 4:1796]
lat = np.array(ds.variables["lat"])
lon = np.array(ds.variables["lon"])

dim = ["x", "y"]
plt.rcParams.update({'font.size': 20})
nvars = 48 + 1
ncolumns = 4
nrows = 6
sequential_cmap = plt.get_cmap("magma", 20)

for i in range(6):    
    for f in range(1, 9):
        plt.figure(figsize=(30, 24))
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = np.log(ds_gefs[0, i, f-1, hrrr_stats_channels.index(var)].clip(min=0))
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-3, vmax=3)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("GEFS Ensemble 0")
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = np.log(ds_hrrr[i, f-1, hrrr_stats_channels.index(var)].clip(min=0))
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-3, vmax=3)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("HRRR Forecast")
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = np.log(ds_prediction[var][0, i, f, :, :].clip(min=0))
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-3, vmax=3)
        plt.title("Pred V3 Ensemble 0")
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = np.log(ds_truth[var][i, f, :, :].clip(min=0))
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-3, vmax=3)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        plt.title("Target")
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_gefs[0, i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-20, vmax=20)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_hrrr[i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-20, vmax=20)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_prediction[var][0, i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-20, vmax=20)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_truth[var][i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=-20, vmax=20)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_gefs[0, i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_hrrr[i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_prediction[var][0, i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_truth[var][i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_gefs[0, i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 14, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_hrrr[i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_prediction[var][0, i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_truth[var][i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_gefs[0, i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 18, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_hrrr[i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_prediction[var][0, i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_truth[var][i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
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
        if i == nvars - 1:
            ax.set_xlabel("longitude")
        else:
            gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_gefs[0, i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        ax = plt.subplot(nrows, ncolumns, 22, projection=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            color="black",
            alpha=0.0,
            draw_labels=True,
            linestyle="None",
        )
        gl.bottom_labels = False
        if f != 0:
            data_to_plot = ds_hrrr[i, f-1, hrrr_stats_channels.index(var)]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        if f != 0:
            data_to_plot = ds_prediction[var][0, i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
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
        if f != 0:
            data_to_plot = ds_truth[var][i, f, :, :]
        else:
            data_to_plot = np.zeros_like(lat)
        im1 = ax.pcolormesh(lon, lat, data_to_plot, cmap=sequential_cmap, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax, shrink=0.5)
        gl.right_labels = False
        gl.top_labels = False
        ax.coastlines(linewidth=0.5, color="white")
        gl.left_labels = False

        plt.suptitle(f"{time[i][:4]}-{time[i][4:6]}-{time[i][6:8]} {time[i][8:10]}:00 | Lead Time: {f*3} Hours", fontsize=35)
        plt.tight_layout()
        plt.savefig(f"./output_movie/reflectivity_movie_{i*9+f}.png")
        plt.close()
        print(f"./output_movie/reflectivity_movie_{i*9+f}.png saved. Time: {ttime.time()-tic}")

import imageio 
import os
writer = imageio.get_writer(f'{output_name}.mp4', fps = 3)
dirFiles = os.listdir('./output_movie/') # list of directory files
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for im in dirFiles:
     writer.append_data(imageio.imread('./output_movie/'+im))
writer.close()
print("saved to twc_mvp1_movie.mp4")
