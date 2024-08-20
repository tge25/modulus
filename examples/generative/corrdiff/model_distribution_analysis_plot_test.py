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

tic=time.time()

names = ["5048320",]


path = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/test/"

for j, name in enumerate(names):
    target = []
    pred = []
    i = 0
    pred = np.load(path+name+f"_{i}.npz")["pred"]
    target = np.load(path+name+f"_{i}.npz")["target"]
    plt.figure(figsize=(12,12))
    plt.subplot(5,2,1)
    plt.imshow(pred[0,4],vmin=0, vmax=1.2)
    plt.subplot(5,2,2)
    plt.imshow(target[0,4],vmin=0, vmax=1.2)

    plt.subplot(5,2,3)
    plt.imshow(pred[0,5],vmin=0, vmax=1.2)
    plt.subplot(5,2,4)
    plt.imshow(target[0,5],vmin=0, vmax=1.2)

    plt.subplot(5,2,5)
    plt.imshow(pred[0,6],vmin=0, vmax=1.2)
    plt.subplot(5,2,6)
    plt.imshow(target[0,6],vmin=0, vmax=1.2)

    plt.subplot(5,2,7)
    plt.imshow(pred[0,7],vmin=0, vmax=1.2)
    plt.subplot(5,2,8)
    plt.imshow(target[0,7],vmin=0, vmax=1.2)

    plt.subplot(5,2,9)
    plt.imshow(pred[0,8],vmin=0, vmax=1.2)
    plt.subplot(5,2,10)
    plt.imshow(target[0,8],vmin=0, vmax=1.2)

    plt.tight_layout()
    plt.savefig("test.png")
    plt.close()