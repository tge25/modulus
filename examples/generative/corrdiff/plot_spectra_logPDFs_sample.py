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

import os
import pickle
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pylab as plt
import xarray
import zarr
from scipy.signal import periodogram

#path_rf = "s3://cwb-diffusions/baselines/rf/era5-cwb-v3/validation_big/samples.nc"
#path_reg = "s3://cwb-diffusions/baselines/regression/era5-cwb-v3/validation_big/samples.nc"
#path_intera5 = "s3://cwb-diffusions/baselines/era5/era5-cwb-v3/validation_big/samples.nc"
#path_resdiff = "s3://cwb-diffusions/generations/era5-cwb-v3/validation_big/samples.zarr"
#baslines_dir = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/baselines/"
path_rf = "image_outdir_entire_val_paper_0.nc"
path_reg = "image_outdir_entire_val_modulus_0.nc"
path_intera5 = "image_outdir_entire_val_clip_mean_flex_18step_overlap4_boundary2_0.nc" 
path_resdiff = "image_outdir_entire_val_patch_learnable_pos_0.nc"

def open_data(file, group=False):
    """Open a NetCDF file and return a dataset, optionally from a group."""
    root = xarray.open_dataset(file)
    root = root.set_coords(["lat", "lon"])
    ds = xarray.open_dataset(file, group=group)
    ds.coords.update(root.coords)
    ds.attrs.update(root.attrs)
    return ds


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two sets of latitude and longitude
    coordinates.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    earth_radius = 6371000
    dlat_rad = lat2_rad - lat1_rad
    dlon_rad = lon2_rad - lon1_rad

    a = (
        np.sin(dlat_rad / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_meters = earth_radius * c
    return distance_meters


def compute_power_spectrum(data, d):
    """Compute the power spectrum of 1D data using Fast Fourier Transform."""
    fft_data = np.fft.fft(data, axis=-2)

    power_spectrum = np.abs(fft_data) ** 2
    power_spectrum /= data.shape[-1] * d
    freqs = np.fft.fftfreq(data.shape[-1], d)
    return freqs, power_spectrum


def average_power_spectrum(data, d):
    """Compute the average power spectrum of 1D data."""
    freqs, power_spectra = periodogram(data, fs=1 / d, axis=-1)

    while power_spectra.ndim > 1:
        power_spectra = power_spectra.mean(axis=0)

    return freqs, power_spectra


def ke_spectra(data):
    """Compute the kinetic energy spectra of wind components."""
    northward_wind_10m = data["northward_wind_10m"]
    eastward_wind_10m = data["eastward_wind_10m"]
    freqs, spec_x = average_power_spectrum(eastward_wind_10m, d=2)
    _, spec_y = average_power_spectrum(northward_wind_10m, d=2)
    spec = spec_x + spec_y
    return freqs, spec


def load_windspeed_dist(data):
    """Load the wind speed distribution from wind components."""
    northward_wind_10m = data["northward_wind_10m"]
    eastward_wind_10m = data["eastward_wind_10m"]
    windspeed_10m = np.sqrt(
        np.multiply(northward_wind_10m, northward_wind_10m)
        + np.multiply(eastward_wind_10m, eastward_wind_10m)
    )
    if isinstance(windspeed_10m, xarray.DataArray):
        windspeed_10m_flat = windspeed_10m.values.flatten()
    else:
        windspeed_10m_flat = windspeed_10m.flatten()
    pdf_values, bin_edges = np.histogram(windspeed_10m_flat, bins="auto", density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, pdf_values


def load_dist(var, bins):
    """Load the distribution of a variable."""
    if isinstance(var, xarray.DataArray):
        flattened_var = var.values.flatten()
    else:
        flattened_var = np.array(var).flatten()
    flattened_var = flattened_var[flattened_var >= 0]
    flattened_var = flattened_var[~np.isnan(flattened_var)]
    pdf_values, bin_edges = np.histogram(flattened_var, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, pdf_values


prediction_rf = open_data(path_rf, group="prediction")
prediction_reg = open_data(path_reg, group="prediction")
prediction_intera5 = open_data(path_intera5, group="prediction")
prediction_resdiff = open_data(path_resdiff, group="prediction") 

wrf = open_data(path_rf, group="truth")
era5 = open_data(path_rf, group="input")
print("data opened")
#prediction_rf = open_data(
#    baslines_dir + "rf/era5-cwb-v3/validation_big/samples.nc", group="prediction"
#)
#prediction_reg = open_data(
#    baslines_dir + "regression/era5-cwb-v3/validation_big/samples.nc",
#    group="prediction",
#)
#prediction_intera5 = open_data(
#    baslines_dir + "era5/era5-cwb-v3/validation_big/samples.nc", group="prediction"
#)

wong_palette = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

fig = plt.figure(figsize=(16, 9))

bins = np.linspace(0, 20, 101)
idx = [0,24,43,64,87,34,1,54,165,145,109,115]
print("target", wrf.maximum_radar_reflectivity)
print("paper", prediction_reg.maximum_radar_reflectivity)
for i in range(12):
    rf_rad_bins, rf_rad_pdf = load_dist(prediction_rf.maximum_radar_reflectivity[:,i], bins)
    reg_rad_bins, reg_rad_pdf = load_dist(prediction_reg.maximum_radar_reflectivity[:,i], bins)
    wrf_rad_bins, wrf_rad_pdf = load_dist(wrf.maximum_radar_reflectivity[i], bins)
    intera5_rad_bins, intera5_rad_pdf = load_dist(prediction_intera5.maximum_radar_reflectivity[:,i], bins)
    resdiff_rad_bins, resdiff_rad_pdf = load_dist(prediction_resdiff.maximum_radar_reflectivity[:,i], bins)
    print(i, "radar pdf done")
    # radar
    plt.subplot(3, 4, i+1)
    plt.plot(
        wrf_rad_bins, np.log(wrf_rad_pdf), label="target", color=wong_palette[4], linewidth=5
    )
    plt.plot(
        rf_rad_bins, np.log(rf_rad_pdf), label="Paper Baseline", color=wong_palette[1], linewidth=1
    )
    plt.plot(
        reg_rad_bins, np.log(reg_rad_pdf), label="Modulus CorrDiff", color=wong_palette[2], linewidth=1
    )
    plt.plot(
        intera5_rad_bins,
        np.log(intera5_rad_pdf),
        label="Patched CorrDiff",
        color=wong_palette[3],
        linewidth=1,
    )
    plt.plot(
        resdiff_rad_bins,
        np.log(resdiff_rad_pdf),
        label="Patched CorrDiff - learnt 100pe",
        color=wong_palette[5],
        linewidth=1,
    )
    if i==0:
        plt.legend()
    plt.xlabel("radar reflectivity (dbz)")
    plt.ylabel("log(PDF)")

plt.tight_layout()
plt.savefig("./distributions_samples.png")
