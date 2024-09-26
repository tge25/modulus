from .hrrr_test import HrrrForecastGEFSDataset
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os


location_hrrr = "/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp"
location_gefs_surface = "/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp"
location_gefs_isobaric = "/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp"


def running_std_mean_t(data):
    mean = np.mean(data, axis=(1,2))
    mean2 = np.mean(data**2, axis=(1,2))
    return mean, mean2

def running_std_mean(mean_list, mean2_list, data):
    temp = np.mean(data, axis=(1,2))
    mean_list = np.append(mean_list, temp[np.newaxis], axis = 0)
    temp = np.mean(data**2, axis=(1,2))
    mean2_list = np.append(mean2_list, temp[np.newaxis], axis = 0)
    return mean_list, mean2_list

if __name__ == "__main__":
    data = HrrrForecastGEFSDataset(train=False)
    i = 0
    while True:       
        d = data[i]
        hrrr = d[0]
        timestep = d[2]  
        precip = hrrr[3]
        snow = hrrr[4]
        ice = hrrr[5]
        freez = hrrr[6]
        rain = hrrr[7]

        if np.mean(precip)>0.1 and (np.mean(ice)>0.005 or np.mean(freez)>0.01):
            print(f"{timestep} | precip mean: {np.mean(precip):.2e}, snow mean: {np.mean(snow):.2e}, ice mean: {np.mean(ice):.2e}, freez mean: {np.mean(freez):.2e}, rain mean: {np.mean(rain):.2e}", flush=True)

        i += 1



        