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
import xskillscore

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

files = ["/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_v3_full1_new_0.nc", "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_v3_full2_new_0.nc"]

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

    plt.figure(figsize=(30,30))
    for lead in range(9):
        samples = {}
        predictions = []
        truths = []
        for file in files:
            prediction = open_data(file, group="prediction")
            predictions.append(prediction.isel(time=slice(0, n_timesteps), ensemble=slice(0, 8), x=slice(0, -1 ,13), y=slice(0, -1 ,13), forecast=lead))
            truth = open_data(file, group="truth")
            truths.append(truth.isel(time=slice(0, n_timesteps), x=slice(0, -1 ,13), y=slice(0, -1 ,13), forecast=lead))

        predictions = xr.concat(predictions, dim="time")
        truths = xr.concat(truths, dim="time")
        
        samples["prediction"] = predictions
        samples["truth"] = truths

        print(samples["prediction"],flush=True)
        print(samples["truth"],flush=True)

        hist = xskillscore.rank_histogram(samples["truth"], samples["prediction"], member_dim="ensemble")

        print("dist done",flush=True)
        print(hist,flush=True)

        '''
        if save_data:
            path = os.path.join(output, "rank_hist.json")
            with open(path, "w") as f:
                json.dump(hist.to_dict(), f)
        '''
        
        for i, field in enumerate(hist):
            print(i, lead, i*9+lead+1)
            plt.subplot(9,9,i*9+lead+1)
            plt.bar(hist['rank'], hist[field])
            plt.title(f"lead time {lead*3:02d} hour - {field}")
        plt.tight_layout()
        savefig("rank_hist")


if __name__ == "__main__":
    app()
