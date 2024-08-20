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
path_rf = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_full1_0.nc"
vars = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain"]

location_gefs_surface = "/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp"
location_gefs_isobaric = "/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp"
        
means_file_surface = os.path.join(location_gefs_surface, 'stats', 'means_surface.npy')
stds_file_surface = os.path.join(location_gefs_surface, 'stats', 'stds_surface.npy')

means_gefs_surface = np.load(means_file_surface)
stds_gefs_surface = np.load(stds_file_surface)

means_file_isobaric = os.path.join(location_gefs_isobaric, 'stats', 'means_isobaric.npy')
stds_file_isobaric = os.path.join(location_gefs_isobaric, 'stats', 'stds_isobaric.npy')

means_gefs_isobaric = np.load(means_file_isobaric)
stds_gefs_isobaric = np.load(stds_file_isobaric)

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
    northward_wind_10m = data["v10m"]
    eastward_wind_10m = data["u10m"]
    freqs, spec_x = average_power_spectrum(eastward_wind_10m, d=3)
    _, spec_y = average_power_spectrum(northward_wind_10m, d=3)
    spec = spec_x + spec_y
    return freqs, spec


def load_windspeed_dist(data):
    """Load the wind speed distribution from wind components."""
    northward_wind_10m = data["v10m"]
    eastward_wind_10m = data["u10m"]
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
wrf = open_data(path_rf, group="truth")
era5 = open_data(path_rf, group="input")

# for mis-unnormalized inferences
for i, var in enumerate(["u10m", "v10m", "t2m"]):
    era5[var] = (era5[var] - stds_gefs_surface[i])/means_gefs_surface[i]
    era5[var] = (era5[var] * stds_gefs_surface[i]) + means_gefs_surface[i]
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

rf_ke_freq, rf_ke_spec = ke_spectra(prediction_rf)
wrf_ke_freq, wrf_ke_spec = ke_spectra(wrf)
era5_ke_freq, era5_ke_spec = ke_spectra(era5)

d = 3  # KM
rf_t2m_freq, rf_t2m_spec = average_power_spectrum(prediction_rf.t2m, d=d)
wrf_t2m_freq, wrf_t2m_spec = average_power_spectrum(wrf.t2m, d=d)
era5_t2m_freq, era5_t2m_spec = average_power_spectrum(era5.t2m, d=d)

print("t2m spectrum done")

rf_rad_freq, rf_rad_spec = average_power_spectrum(
    prediction_rf.precip, d=d
)
wrf_rad_freq, wrf_rad_spec = average_power_spectrum(wrf.precip, d=d)

print("radar spectrum done")

rf_windspeed_bins, rf_windspeed_pdf = load_windspeed_dist(prediction_rf)
wrf_windspeed_bins, wrf_windspeed_pdf = load_windspeed_dist(wrf)
era5_windspeed_bins, era5_windspeed_pdf = load_windspeed_dist(era5)

print("wind done")

rf_t2m_bins, rf_t2m_pdf = load_dist(prediction_rf.t2m, 30)
wrf_t2m_bins, wrf_t2m_pdf = load_dist(wrf.t2m, 30)
era5_t2m_bins, era5_t2m_pdf = load_dist(era5.t2m, 30)

print("t2m pdf done")

bins = np.linspace(0, 5, 101)
rf_rad_bins, rf_rad_pdf = load_dist(prediction_rf.precip, bins)
wrf_rad_bins, wrf_rad_pdf = load_dist(wrf.precip, bins)

print("radar pdf done")

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
# KE
ax0 = plt.subplot(231)
ax0.annotate("(a)", xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=12)
ax0.loglog(wrf_ke_freq, wrf_ke_spec, label="HRRR", color=wong_palette[4], linewidth=5)
ax0.loglog(rf_ke_freq, rf_ke_spec, label="Prediction", color=wong_palette[1], linewidth=1)
ax0.loglog(era5_ke_freq, era5_ke_spec, label="GEFS", color=wong_palette[0], linewidth=1)

ax0.set_xlabel("Zonal wavenumber (1/km)")
ax0.set_ylabel(r"Kinetic energy spectra ($\mathrm{m^2/s^2}$)")
ax0.set_ylim(bottom=1e-1)
ax0.spines["right"].set_visible(False)
ax0.spines["top"].set_visible(False)
ax0.legend()


ax1 = plt.subplot(232)
ax1.annotate("(b)", xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=12)
ax1.loglog(wrf_t2m_freq, wrf_t2m_spec, label="wrf", color=wong_palette[4], linewidth=5)
ax1.loglog(rf_t2m_freq, rf_t2m_spec, label="rf", color=wong_palette[1], linewidth=1)
ax1.loglog(
    era5_t2m_freq, era5_t2m_spec, label="era5", color=wong_palette[0], linewidth=1
)
ax1.set_xlabel("Zonal wavenumber (1/km)")
ax1.set_ylabel("2m temperature spectra (K)")
ax1.set_ylim(bottom=1e-1)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)

# radar
ax2 = plt.subplot(233)
ax2.annotate("(c)", xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=12)
ax2.loglog(wrf_rad_freq, wrf_rad_spec, label="HRRR", color=wong_palette[4], linewidth=5)
ax2.loglog(rf_rad_freq, rf_rad_spec, label="Prediction", color=wong_palette[1], linewidth=1)
ax2.set_xlabel("Zonal wavenumber (1/km)")
ax2.set_ylabel("radar reflectivity spectra (dbz)")
ax2.set_ylim(bottom=1e-1)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
#ax2.legend()

# windspeed
ax4 = plt.subplot(234)
ax4.annotate("(d)", xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=12)
ax4.plot(
    wrf_windspeed_bins,
    np.log(wrf_windspeed_pdf),
    label="wrf",
    color=wong_palette[4],
    linewidth=5,
)
ax4.plot(
    rf_windspeed_bins,
    np.log(rf_windspeed_pdf),
    label="RF",
    color=wong_palette[1],
    linewidth=1,
)
ax4.plot(
    era5_windspeed_bins,
    np.log(era5_windspeed_pdf),
    label="era5",
    color=wong_palette[0],
    linewidth=1,
)
ax4.set_xlabel("10 meter windspeed (m/s)")
ax4.set_ylabel("log(PDF)")
ax4.set_xlim([0, 30])
ax4.spines["right"].set_visible(False)
ax4.spines["top"].set_visible(False)

ax5 = plt.subplot(235)
ax5.annotate("(e)", xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=12)
ax5.plot(
    wrf_t2m_bins, np.log(wrf_t2m_pdf), label="wrf", color=wong_palette[4], linewidth=5
)
ax5.plot(
    rf_t2m_bins, np.log(rf_t2m_pdf), label="rf", color=wong_palette[1], linewidth=1
)
ax5.plot(
    era5_t2m_bins,
    np.log(era5_t2m_pdf),
    label="era5",
    color=wong_palette[0],
    linewidth=1,
)
ax5.set_xlabel("2 meter temperature (K)")
ax5.set_ylabel("log(PDF)")
ax5.spines["right"].set_visible(False)
ax5.spines["top"].set_visible(False)

# radar
ax6 = plt.subplot(236)
ax6.annotate("(f)", xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=12)

ax6.plot(
    wrf_rad_bins, np.log(wrf_rad_pdf), label="wrf", color=wong_palette[4], linewidth=5
)
ax6.plot(
    rf_rad_bins, np.log(rf_rad_pdf), label="rf", color=wong_palette[1], linewidth=1
)
ax6.set_xlabel("radar reflectivity (dbz)")
ax6.set_ylabel("log(PDF)")
ax6.spines["right"].set_visible(False)
ax6.spines["top"].set_visible(False)


plt.tight_layout()
plt.savefig("./spectra_and_distributions_twc_mvp1.png")
