import zarr
import numpy as np
import pandas as pd
import xarray as xr
import pygrib
import os
import time

# Define a function to read grid data
def grid_read(timestamp, lead_time):
    time_str = timestamp.strftime('%Y%m%d')
    path = os.path.join(source_dir, f"hrrr.{time_str}", "conus", f"hrrr.t{timestamp.strftime('%H')}z.wrfsfcf{lead_time:02d}.grib2")
    grbs = pygrib.open(path)
    return grbs

start_date = "2020-12-02"
end_date = "2024-07-31"
timestamps = pd.date_range(start=start_date, end=end_date, freq='6H')

source_dir = "/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp/data"
target_path = '/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp/HRRR_forecasts_2023.zarr'
store = zarr.DirectoryStore(target_path)
tic = time.time()

# Initialize the Zarr store if it doesn't exist
grbs = grid_read(timestamps[0], 0)
lat, lon = grbs[1].latlons()
x = np.arange(lat.shape[0])
y = np.arange(lat.shape[1])

# Check if the Zarr store already exists
if os.path.exists(target_path):
    # Open the existing Zarr store
    ds = xr.open_zarr(store, consolidated=True)
    # Get the last timestamp in the Zarr store
    print(ds.time.values[-1])
    print(ds.time.values[-1][:-3])
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
    ).chunk(chunks={"x": 1059, "y": 1799})

    data_dict["lon"] = xr.DataArray(
        lon,
        dims=["x", "y"],
        coords={"x": x, "y": y},
    ).chunk(chunks={"x": 1059, "y": 1799})
    
    # Initialize the Zarr store
    ds = xr.Dataset(data_dict)
    # Add additional metadata to the Dataset
    ds.attrs['title'] = 'HRRR dataset for NV-TWC MVP project'
    ds.attrs['description'] = 'HRRR forecasts dataset with up to 24 hours lead time from 2020/12/01 to 2024/07/31.'
    ds.to_zarr(store, mode='w', consolidated=True)
    last_time = pd.to_datetime(start_date)
    creating_file = True

# Continue processing from the last timestamp
timestamps = pd.date_range(start=last_time, end=end_date, freq='6H')
channels = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "refc"]
source_channels = ["10 metre U wind component", "10 metre V wind component", "Temperature", "Total Precipitation", "Categorical snow", "Categorical ice pellets", "Categorical freezing rain", "Categorical rain", "Maximum/Composite radar reflectivity"]
precip_channels = ["Total Precipitation", "Categorical snow", "Categorical ice pellets", "Categorical freezing rain", "Categorical rain"]

data_list = []
count = 0
data_append_count = 0
for timestamp in timestamps:
    for lead_time in range(0, 25, 3):
        src_time_str = timestamp.strftime('%Y%m%d%H')
        date_str = src_time_str + f"f{lead_time:02d}"
        try:
            grbs = grid_read(timestamp, lead_time)
        except OSError:
            print(f"file {timestamp} lead time {lead_time} not found")
            continue
        values = []
        for channel in source_channels:
            if channel in precip_channels and lead_time == 0:
                grbs_prev = grid_read(timestamp - pd.Timedelta(hours=1), 1)
                grb = grbs_prev.select(name=channel)[-1]
                values.append(grb["values"])
            else:
                grb = grbs.select(name=channel)[-1]
                values.append(grb["values"])
        values = np.stack(values)

        # Add new axis for time dimension
        values = np.expand_dims(values, axis=0)

        # Create a DataArray for the values with time dimension
        data_array = xr.DataArray(
            values,
            dims=["time", "channel", "x", "y"],
            coords={"time": [date_str], "channel": channels, "x": x, "y": y},
        ).chunk({"time": 1, "channel": len(channels), "x": 1059, "y": 1799})
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
