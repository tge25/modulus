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

import os

import matplotlib.pyplot as plt
import json
import numpy as np
import typer
import xarray
import xarray as xr
from scipy.fft import irfft
from scipy.signal import periodogram
import torch
import numpy as np
import time
def open_data(file, group=False):
    """
    Opens a dataset from a NetCDF file.

    Parameters:
        file (str): Path to the NetCDF file.
        group (bool, optional): Whether to open the file as a group. Default is False.

    Returns:
        xarray.Dataset: An xarray dataset containing the data from the NetCDF file.
    """
    root = xarray.open_dataset(file)
    root = root.set_coords(["lat", "lon"])
    ds = xarray.open_dataset(file, group=group)
    ds.coords.update(root.coords)
    ds.attrs.update(root.attrs)

    return ds

app = typer.Typer(pretty_exceptions_show_locals=False)

tic = time.time()
vars = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain"]
ds_hrrr = xr.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp_valid/HRRR_forecasts_2024.zarr", consolidated=True)
ds_gefs = xr.open_zarr("/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp_ens_valid/GEFS_surface_2024_winter_storm.zarr", consolidated=True)

ds_gefs['x'] = ds_gefs.lat.values[:,0]
ds_gefs['y'] = ds_gefs.lon.values[0,:]%360
lat = ds_hrrr.lat
lon = ds_hrrr.lon%360
ds_gefs = ds_gefs.interp(x=lat, y=lon)
ds_gefs = ds_gefs['values'].values
ds_hrrr = ds_hrrr['values'].values

@app.command()
def main(output, plot=True, save_data=True, n_ensemble: int = -1, n_timesteps: int = -1):
    """
    Generate and save multiple power spectrum plots from input data.

    Parameters:
        file (str): Path to the input data file.
        output (str): Directory where the generated plots will be saved.

    This function loads and processes various datasets from the input file,
    calculates their power spectra, and generates and saves multiple power spectrum plots.
    The plots include kinetic energy, temperature, and reflectivity power spectra.
    """
    os.makedirs(output, exist_ok=True)

    def savefig(name):
        path = os.path.join(output, name + ".png")
        plt.savefig(path)

    print("n_ensemble", n_ensemble, "n_timesteps", n_timesteps)
    n_members = 31
    
    plt.figure(figsize=(40,30))
    for lead in range(9):
        predictions = ds_gefs
        truths = ds_hrrr

        print(lead, time.time()-tic,flush=True)

        hist = torch.zeros((8, n_members+1))
        with torch.no_grad():        
            for idx, var in enumerate(vars):  
                prediction = torch.from_numpy(predictions[:,:,:,idx,::2,::2]).cuda()
                truth = torch.from_numpy(truths[:,:,idx,::2,::2]).cuda()      
                sorted_predictions, _ = torch.sort(prediction, dim=0)
                for member in range(n_members+1):
                    if member==0:
                        a = -float('inf')
                        b = sorted_predictions[0]
                    elif member==n_members:
                        a = sorted_predictions[-1]
                        b = float('inf')
                    else:
                        a = sorted_predictions[member-1]
                        b = sorted_predictions[member]
                    hist[idx, member] = torch.sum((truth>=a) & (truth<b)).detach()/truth.numel()
                    print(idx, member, torch.sum((truth>=a) & (truth<b))/truth.numel(), flush=True)

        hist = hist.cpu().detach().numpy()

        print(lead, time.time()-tic,flush=True)
        for i in range(8):
            plt.subplot(9,9,i*9+lead+1)
            plt.bar(np.arange(n_members+1),hist[i])
            plt.title(f"lead time {lead*3:02d} hour - {vars[i]}")
        plt.tight_layout()
        savefig("rank_hist")


if __name__ == "__main__":
    app()
