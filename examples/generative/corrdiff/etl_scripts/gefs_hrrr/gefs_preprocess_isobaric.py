# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 23:22:54 2024

@author: getao
"""

import zarr
import numpy as np
import pandas as pd
import xarray as xr
import pygrib
import os
import time
# Define a function to read grid data

def grid_read(timestamp, lead_time, res=25):
    time_str = timestamp.strftime('%Y%m%d')
    hour = timestamp.strftime('%H')
    if res==25:
        path = source_dir+f"gefs.{time_str}/gec00.t{hour}z.pgrb2s.0p25.f0{lead_time:02d}"
    elif res==50:
        path = source_dir+f"gefs.{time_str}/gec00.t{hour}z.pgrb2a.0p50.f0{lead_time:02d}"      
    grbs = pygrib.open(path)
    return grbs


for year in range(2020, 2025):
    # split each year into two datasets
    for i in range(0,2):
        if year==2020:
            i = 1
            start_date = f"{year}-12-02"
            end_date = f"{year+1}-01-01"
            fstr = ""
        elif year==2024:
            i = 1
            start_date = f"{year}-01-01"
            end_date = f"{year}-08-01"
            fstr = ""
        elif i==0:
            start_date = f"{year}-01-01"
            end_date = f"{year}-7-01"
            fstr = "_01_06"
        else:
            start_date = f"{year}-07-01"
            end_date = f"{year+1}-01-01"
            fstr = "_07_12"
        timestamps = pd.date_range(start=start_date, end=end_date, freq='6H')
        
        source_dir = "/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp/data/"
        target_path = f'/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp/GEFS_isobaric_{year}{fstr}.zarr'
        store = zarr.DirectoryStore(target_path)
        tic = time.time()
        
        # Initialize the Zarr store if it doesn't exist
        grbs = grid_read(timestamps[0], 0, res=50)
        lat, lon = grbs[1].latlons()
        x = np.arange(lat.shape[0])
        y = np.arange(lat.shape[1])
            
        # Check if the Zarr store already exists
        if os.path.exists(target_path):
            # Open the existing Zarr store
            ds = xr.open_zarr(store, consolidated=True)
            # Get the last timestamp in the Zarr store
            last_time = pd.to_datetime(ds.time.values[-1][:-3], format='%Y%m%d%H')
            start_date = last_time.strftime('%Y-%m-%d %H:%M:%S')
            creating_file = False
        else:
            # Create an xarray Dataset with the 'time' dimension
            data_dict = {}
            data_dict["lat"] = xr.DataArray(
                lat,
                dims=["x", "y"],
                coords={"x": x, "y": y},
            ).chunk(chunks={"x": 721, "y": 1440})
        
            data_dict["lon"] = xr.DataArray(
                lon,
                dims=["x", "y"],
                coords={"x": x, "y": y},
            ).chunk(chunks={"x": 721, "y": 1440})
            
            # Initialize the Zarr store
            ds = xr.Dataset(data_dict)
            # Add additional metadata to the Dataset
            ds.attrs['title'] = 'GEFS dataset for NV-TWC MVP project'
            ds.attrs['description'] = 'GEFS forecasts dataset with up to 24 hours lead time from 2020/12/01 to 2024/07/31.'
            ds.to_zarr(store, mode='w', consolidated=True)
            last_time = pd.to_datetime(start_date)
            creating_file = True
        
        # Continue processing from the last timestamp
        timestamps = pd.date_range(start=last_time, end=end_date, freq='6H')
        surface_channels = ["u10m", "v10m", "t2m", "q2m", "sp", "msl", "precipitable_water"]
        surface_source_channels = ["10 metre U wind component", "10 metre V wind component", "2 metre temperature", "2 metre relative humidity", "Surface pressure", "Pressure reduced to MSL", "Precipitable water"]
        level_source_channels = ["U component of wind", "V component of wind", "Geopotential height", "Temperature", "Relative humidity"]
        channels = ["u", "v", "z", "t", "q"]
        pressures = ["1000", "925", "850", "700", "500", "250"]
        level_channels = [channel+pressure for channel in channels for pressure in pressures]
        levels = [-1,-2,-3,-4,-5,-8]
        
        data_list = []
        count = 0
        data_append_count = 0
        for timestamp in timestamps:
            for lead_time in range(0, 25, 3):
                src_time_str = timestamp.strftime('%Y%m%d%H')
                date_str = src_time_str + f"f{lead_time:02d}"
                try:
                    grbs = grid_read(timestamp, lead_time, res=50)
                except OSError:
                    print(f"file {timestamp} lead time {lead_time} not found")
                    continue
                values = []
                for channel in level_source_channels:
                    for level in levels:
                        grb = grbs.select(name=channel)[level]
                        values.append(grb["values"])
                values = np.stack(values)
        
                # Add new axis for time dimension
                values = np.expand_dims(values, axis=0)
        
                # Create a DataArray for the values with time dimension
                data_array = xr.DataArray(
                    values,
                    dims=["time", "channel", "x", "y"],
                    coords={"time": [date_str], "channel": level_channels, "x": x, "y": y},
                ).chunk({"time": 1, "channel": len(level_channels), "x": 721, "y": 1440})
                # Convert DataArray to Dataset to append
                data_list.append(xr.Dataset({"values": data_array}))
                data_append_count += 1
        
                if data_append_count % 36 == 0 or (timestamp==timestamps[-1] and lead_time==24):
                    data_append_count = 0
                    data_list = xr.concat(data_list, dim='time')
                    # Save the dataset to a Zarr store with consolidated metadata
                    if creating_file and count==0:
                        data_list.to_zarr(store, mode='a', consolidated=True)
                    else:
                        data_list.to_zarr(store, consolidated=True, append_dim="time")
                    data_list = []
                    count += 1
                    print(f"{date_str} saved to {target_path}. Elapsed time {time.time()-tic} seconds. ", flush=True)
        
        # Optionally, verify the consolidated metadata
        with xr.open_zarr(target_path, consolidated=True) as ds:
            print(ds)
            print(list(ds.channel.values))
            print(ds.values)
            print(ds.lat)
            print(ds.lon)
