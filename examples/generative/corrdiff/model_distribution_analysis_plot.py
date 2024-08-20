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

names = ["5120",
        "107520",
        "158720",
        "209920",
        "517120",
        "1034240",
        "1546240",
        "2012160",
        "3041280",
        "4019200",
        "5048320",]


path = "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/test/"

for j, name in enumerate(names):
    target = []
    pred = []
    for i in range(8):
        pred.append(np.load(path+name+f"_{i}.npz")["pred"][:,:4].flatten()[::37])
        target.append(np.load(path+name+f"_{i}.npz")["target"][:,:4].flatten()[::37])   
    pred = np.concatenate(pred, axis=0).flatten()
    target = np.concatenate(target, axis=0).flatten()
    plt.figure(figsize=(11,8))
    plt.subplot(2,3,1)
    data = np.random.normal(0, 1, pred.size)
    plt.hist(data, bins=100, log=True) 
    plt.title("added noise")
    plt.xlim(-6, 6)
    plt.ylim(1, 5e6)
    plt.subplot(2,3,2)
    plt.hist(pred, bins=100, range=(-30, 100), log=True)  
    plt.title("denoiser output - real number (residual) channels")
    plt.xlim(-30, 100)
    plt.ylim(1, 5e7)
    plt.subplot(2,3,3)
    plt.hist(target, bins=100, range=(-30, 100), log=True)  
    plt.title("target - real number (residual) channels")
    plt.xlim(-30, 100)
    plt.ylim(1, 5e7)

    target = []
    pred = []
    for i in range(8):
        pred.append(np.load(path+name+f"_{i}.npz")["pred"][:,4:].flatten()[::37])
        target.append(np.load(path+name+f"_{i}.npz")["target"][:,4:].flatten()[::37])    
    pred = np.concatenate(pred, axis=0).flatten()
    target = np.concatenate(target, axis=0).flatten()
    plt.subplot(2,3,4)
    data = np.random.normal(0, 1, pred.size)
    plt.hist(data, bins=100, log=True) 
    plt.title("added noise")
    plt.xlim(-6, 6)
    plt.ylim(1, 5e6)
    plt.subplot(2,3,5)
    plt.hist(pred, bins=100, log=True)  
    plt.title("denoiser output - binary channels")
    plt.xlim(-0.1, 1.1)
    plt.ylim(1, 5e7)
    plt.subplot(2,3,6)
    plt.hist(target, bins=100, log=True)  
    plt.title("target - binary channels")
    plt.xlim(-0.1, 1.1)
    plt.ylim(1, 5e7)
    plt.suptitle(f"{int(name)/1000} kimg")
    plt.tight_layout()
    plt.savefig("./output_movie/reflectivity_movie%d.png"%j)
    plt.close()
    print(j, time.time()-tic, flush=True)

import imageio 
import os
writer = imageio.get_writer('twc_mvp1_denoiser_distribution.mp4', fps = 3)
dirFiles = os.listdir('./output_movie/') #list of directory files
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for im in dirFiles:
     writer.append_data(imageio.imread('./output_movie/'+im))
writer.close()