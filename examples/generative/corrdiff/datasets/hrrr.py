# Data loader for TWC MVP: GEFS and HRRR forecasts
# adapted from https://gitlab-master.nvidia.com/earth-2/corrdiff-internal/-/blob/dpruitt/hrrr/explore/dpruitt/hrrr/datasets/hrrr.py

from datetime import datetime, timedelta
import glob
import logging
import os
from typing import Iterable, Tuple, Union
import cv2
import s3fs

import dask
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
import xarray as xr
from .time import time_range
from modulus.distributed import DistributedManager

from .base import ChannelMetadata, DownscalingDataset

'''
TO DO LIST:
missing samples file
mean and std
'''



import nvtx

hrrr_stats_channels = ["u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain", "refc"]
gefs_surface_channels = ["u10m", "v10m", "t2m", "q2m", "sp", "msl", "precipitable_water"]
gefs_isobaric_channels = ['u1000', 'u925', 'u850', 'u700', 'u500', 'u250', 'v1000', 'v925', 'v850', 'v700', 'v500', 'v250', 'z1000', 'z925', 'z850', 'z700', 'z500', 'z250', 't1000', 't925', 't850', 't700', 't500', 't250',  'q1000', 'q925', 'q850', 'q700', 'q500', 'q250']

def get_dataset(*, train, **kwargs):
    return HrrrForecastGEFSDataset(
        **kwargs,
        train=train
    )


class HrrrForecastGEFSDataset(DownscalingDataset):
    '''
    Paired dataset object serving time-synchronized pairs of GEFS (surface and isobaric) and HRRR samples
    Expects data to be stored under directory specified by 'location'
        GEFS under <root_dir>/gefs/
        HRRR under <root_dir>/hrrr/
    '''
    def __init__(
        self,
        *,
        location_hrrr: str = "/lustre/fsw/coreai_climate_earth2/datasets/hrrr_forecast/twc_mvp",
        location_gefs_surface: str = "/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp",
        location_gefs_isobaric: str = "/lustre/fsw/coreai_climate_earth2/datasets/gefs/twc_mvp",
        train: bool = True,
        normalize: bool = True,
        dataset_name: str = None,
        hrrr_stats_dir: str = 'stats',
        exclude_gefs_surface_channels: Iterable[str] = (),
        exclude_gefs_isobaric_channels: Iterable[str] = (),
        exclude_hrrr_channels: Iterable[str] = (),
        out_channels: Iterable[str] = ("u10m", "v10m", "t2m", "precip", "cat_snow", "cat_ice", "cat_freez", "cat_rain"),
        gefs_surface_channels: Iterable[str] = ("u10m", "v10m", "t2m", "q2m", "sp", "msl", "precipitable_water"),
        gefs_isobaric_channels: Iterable[str] = ('u1000', 'u925', 'u850', 'u700', 'u500', 'u250', 'v1000', 'v925', 'v850', 'v700', 'v500', 'v250', 'z1000', 'z925', 'z850', 'z700', 'z500', 'z250', 't1000', 't925', 't850', 't700', 't500', 't250',  'q1000', 'q925', 'q850', 'q700', 'q500', 'q250'),
        train_years: Iterable[int] = (2020, 2021, 2022, 2023),
        valid_years: Iterable[int] = (2024,),
        hrrr_window: Union[Tuple[Tuple[int, int], Tuple[int, int]], None] = None,
        sample_shape: Tuple[int,int] = None,
        ds_factor: int = 1,
        shard: bool = False,
        overfit: bool = False,
        use_all: bool = False,
    ):
        dask.config.set(scheduler='synchronous') # for threadsafe multiworker dataloaders
        self.location_hrrr = location_hrrr
        self.location_gefs_surface = location_gefs_surface
        self.location_gefs_isobaric = location_gefs_isobaric
        self.train = train
        self.normalize = normalize
        self.dataset_name = dataset_name
        self.exclude_gefs_surface_channels = list(exclude_gefs_surface_channels)
        self.exclude_gefs_isobaric_channels = list(exclude_gefs_isobaric_channels)
        self.exclude_hrrr_channels = list(exclude_hrrr_channels)
        self.hrrr_channels = out_channels
        self.gefs_surface_channels = gefs_surface_channels
        self.gefs_isobaric_channels = gefs_isobaric_channels
        self.train_years = list(train_years)
        self.valid_years = list(valid_years)
        self.hrrr_window = hrrr_window
        self.sample_shape = sample_shape
        self.ds_factor = ds_factor
        self.shard = shard
        self.use_all = use_all
        self.s3 = s3fs.S3FileSystem() if "s3:" in location_hrrr else None

        self._get_files_stats()
        self.overfit = overfit

        self.kept_hrrr_channel_names = self._get_hrrr_channel_names()
        kept_hrrr_channels = [hrrr_stats_channels.index(x) for x in self.kept_hrrr_channel_names]
        means_file = os.path.join(self.location_hrrr, 'stats', 'means.npy')
        stds_file = os.path.join(self.location_hrrr, 'stats', 'stds.npy')
        if self.s3:
            print("loading stats from s3")
            means_file = self.s3.open(means_file)
            stds_file = self.s3.open(stds_file)


        self.means_hrrr = np.load(means_file)[kept_hrrr_channels, None, None]
        self.stds_hrrr = np.load(stds_file)[kept_hrrr_channels, None, None]


        self.kept_gefs_surface_channel_names = self._get_gefs_surface_channel_names()
        kept_gefs_surface_channels = [gefs_surface_channels.index(x) for x in self.kept_gefs_surface_channel_names]
        means_file_surface = os.path.join(self.location_gefs_surface, 'stats', 'means_surface.npy')
        stds_file_surface = os.path.join(self.location_gefs_surface, 'stats', 'stds_surface.npy')
        if self.s3:
            means_file_surface = self.s3.open(means_file_surface)
            stds_file_surface = self.s3.open(stds_file_surface)

        self.means_gefs_surface = np.load(means_file_surface)[kept_gefs_surface_channels, None, None]
        self.stds_gefs_surface = np.load(stds_file_surface)[kept_gefs_surface_channels, None, None]


        self.kept_gefs_isobaric_channel_names = self._get_gefs_isobaric_channel_names()
        kept_gefs_isobaric_channels = [gefs_isobaric_channels.index(x) for x in self.kept_gefs_isobaric_channel_names]
        means_file_isobaric = os.path.join(self.location_gefs_isobaric, 'stats', 'means_isobaric.npy')
        stds_file_isobaric = os.path.join(self.location_gefs_isobaric, 'stats', 'stds_isobaric.npy')
        if self.s3:
            means_file_isobaric = self.s3.open(means_file_isobaric)
            stds_file_isobaric = self.s3.open(stds_file_isobaric)

        self.means_gefs_isobaric = np.load(means_file_isobaric)[kept_gefs_isobaric_channels, None, None]
        self.stds_gefs_isobaric = np.load(stds_file_isobaric)[kept_gefs_isobaric_channels, None, None]


    def _get_hrrr_channel_names(self):
        if self.hrrr_channels:
            kept_hrrr_channels = [x for x in self.hrrr_channels if x in self.base_hrrr_channels]
            if len(kept_hrrr_channels) != len(self.hrrr_channels):
                print(f'Not all HRRR channels in dataset. Missing {self.hrrr_channels-kept_hrrr_channels}')
        else:
            kept_hrrr_channels = self.base_hrrr_channels

        return list(kept_hrrr_channels)

    def _get_gefs_surface_channel_names(self):
        if self.gefs_surface_channels:
            kept_gefs_surface_channels = [x for x in self.gefs_surface_channels if x in self.base_gefs_surface_channels]
            if len(kept_gefs_surface_channels) != len(self.gefs_surface_channels):
                print(f'Not all GEFS surface channels in dataset. Missing {self.gefs_surface_channels-kept_gefs_surface_channels}')
        else:
            kept_gefs_surface_channels = self.base_gefs_surface_channels

        return list(kept_gefs_surface_channels)

    def _get_gefs_isobaric_channel_names(self):
        if self.gefs_isobaric_channels:
            kept_gefs_isobaric_channels = [x for x in self.gefs_isobaric_channels if x in self.base_gefs_isobaric_channels]
            if len(kept_gefs_isobaric_channels) != len(self.gefs_isobaric_channels):
                print(f'Not all GEFS isobaric channels in dataset. Missing {self.gefs_isobaric_channels-kept_gefs_isobaric_channels}')
        else:
            kept_gefs_isobaric_channels = self.base_gefs_isobaric_channels

        return list(kept_gefs_isobaric_channels)

    def _get_files_stats(self):
        '''
        Scan directories and extract metadata for GEFS (surface and isobaric) and HRRR

        Note: This makes the assumption that the lowest numerical year has the 
        correct channel ordering for the means
        '''

        # GEFS surface parsing
        self.ds_gefs_surface = {}
        if self.s3:
            print("initializing input from s3")
            gefs_surface_short_paths = self.s3.glob(os.path.join(self.location_gefs_surface.replace("s3://",""),'gefs_surface',"????.zarr"))
            gefs_surface_paths = ["s3://" + path for path in gefs_surface_short_paths]
        else:
            gefs_surface_paths = glob.glob(os.path.join(self.location_gefs_surface, "*surface*.zarr"), recursive=True)
        gefs_surface_years = [os.path.basename(x).replace('.zarr', '')[-10:] for x in gefs_surface_paths if "stats" not in x]
        self.gefs_surface_paths = dict(zip(gefs_surface_years, gefs_surface_paths))

        # keep only training or validation years
        years = self.train_years if self.train else self.valid_years
        self.gefs_surface_paths = {year: path for (year, path) in self.gefs_surface_paths.items() if int(year[:4]) in years}
        self.n_years_surface = len(self.gefs_surface_paths)
        first_key = list(self.gefs_surface_paths.keys())[0]

        with xr.open_zarr(self.gefs_surface_paths[first_key], consolidated=True) as ds:
            self.base_gefs_surface_channels = list(ds.channel.values)
            self.gefs_surface_lat = ds.lat
            self.gefs_surface_lon = ds.lon%360

        self.ds_gefs_isobaric = {}
        gefs_isobaric_paths = [path.replace("surface", "isobaric") for path in gefs_surface_paths]
        gefs_isobaric_years = [os.path.basename(x).replace('.zarr', '')[-10:] for x in gefs_isobaric_paths if "stats" not in x]
        self.gefs_isobaric_paths = dict(zip(gefs_isobaric_years, gefs_isobaric_paths))

        # keep only training or validation years
        years = self.train_years if self.train else self.valid_years
        self.gefs_isobaric_paths = {year: path for (year, path) in self.gefs_isobaric_paths.items() if int(year[:4]) in years}
        self.n_years_surface = len(self.gefs_isobaric_paths)
        first_key = list(self.gefs_isobaric_paths.keys())[0]

        #with xr.open_zarr(gefs_isobaric_paths[0], consolidated=True) as ds:
        with xr.open_zarr(self.gefs_isobaric_paths[first_key], consolidated=True) as ds:
            self.base_gefs_isobaric_channels = list(ds.channel.values)
            self.gefs_isobaric_lat = ds.lat
            self.gefs_isobaric_lon = ds.lon%360

        # HRRR parsing
        self.ds_hrrr = {}
        if self.s3:
            print("initializing output from s3")
            hrrr_short_paths = self.s3.glob(os.path.join(self.location_hrrr.replace("s3://",""),"????.zarr"))
            hrrr_paths = ["s3://" + path for path in hrrr_short_paths]
        else:
            hrrr_paths = glob.glob(os.path.join(self.location_hrrr, "*[!backup].zarr"), recursive=True)

        hrrr_years = [os.path.basename(x).replace('.zarr', '')[-10:] for x in hrrr_paths if "stats" not in x]
        self.hrrr_paths = dict(zip(hrrr_years, hrrr_paths))
        # keep only training or validation years
        self.hrrr_paths = {year: path for (year, path) in self.hrrr_paths.items() if int(year[:4]) in years}
        self.years = set([int(key[:4]) for key in self.hrrr_paths.keys()])

        #assert set(gefs_surface_years) == set(hrrr_years) == set(gefs_isobaric_years), 'Number of years for GEFS surface, GEFS isobaric, and HRRR must match'
        with xr.open_zarr(hrrr_paths[0], consolidated=True) as ds:
            self.base_hrrr_channels = list(ds.channel.values)
            self.hrrr_lat = ds.lat
            self.hrrr_lon = ds.lon%360
        
        if self.hrrr_window is None:
            self.hrrr_window = ((0, self.hrrr_lat.shape[0]), (0, self.hrrr_lat.shape[1]))
        
        self.n_samples_total = self.compute_total_samples()

    def __len__(self):
        return len(self.valid_samples)

    def crop_to_fit(self, x):
        '''
        Crop HRRR to get nicer dims
        '''
        ((y0, y1), (x0, x1)) = self._get_crop_box()
        return x[..., y0:y1, x0:x1]

    def to_datetime(self, date):
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                     / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)

    def compute_total_samples(self):

        #Loop through all years and count the total number of samples

        first_year = min(self.years)
        last_year = max(self.years)
        if first_year == 2020:
            first_sample = datetime(year=2020, month=12, day=2, hour=0, minute=0, second=0)
        else:
            first_sample = datetime(year=first_year, month=1, day=1, hour=0, minute=0, second=0)
        if last_year == 2024:
            last_sample = datetime(year=2024, month=7, day=31, hour=19, minute=0, second=0)
        else:
            last_sample = datetime(year=last_year, month=12, day=31, hour=19, minute=0, second=0)
        
        logging.info("First sample is {}".format(first_sample)) 
        logging.info("Last sample is {}".format(last_sample))

        
        all_datetimes = time_range(first_sample, last_sample, step=timedelta(hours=6), inclusive=True)

        all_datetimes = set(dt for dt in all_datetimes if dt.year in self.years)       
        all_datetimes = set(time.strftime('%Y%m%d%H')+f"f{f:02d}" for time in all_datetimes for f in range(0,25,3))

        samples_file = os.path.join(self.location_hrrr, 'missing_samples.npy')
        if self.s3:
            samples_file = self.s3.open(samples_file)


        # missing samples file
        missing_samples = np.load(samples_file, allow_pickle=True)
        missing_samples = set(missing_samples) #hash for faster lookup
        self.valid_samples = sorted(all_datetimes.difference(missing_samples)) # exclude missing samples
        logging.info('Total datetimes in training set are {} of which {} are valid'.format(len(all_datetimes), len(self.valid_samples)))

        if self.shard: # use only part of dataset in each training process
            dist_manager = DistributedManager()
            self.valid_samples = np.array_split(self.valid_samples, dist_manager.world_size)[dist_manager.rank]

        return len(self.valid_samples)


    def normalize_input(self, x, means, stds):
        x = x.astype(np.float32)
        if self.normalize:
            x -= means
            x /= stds
        return x

    def denormalize_input(self, x):
        x = x.astype(np.float32)
        if (len(x.shape)==3) and self.normalize:
            x[:7] *= self.means_gefs_surface
            x[:7] += self.stds_gefs_surface
            x[7:] *= self.means_gefs_isobaric
            x[7:] += self.stds_gefs_isobaric
        elif (len(x.shape)==4) and self.normalize:
            x[:,:7] *= self.means_gefs_surface[None]
            x[:,:7] += self.stds_gefs_surface[None]
            x[:,7:] *= self.means_gefs_isobaric[None]
            x[:,7:] += self.stds_gefs_isobaric[None]
        return x

    def _get_gefs_surface(self, ts, lat, lon):
        '''
        Retrieve GEFS surface samples from zarr files
        '''
        gefs_surface_handle = self._get_ds_handles(self.ds_gefs_surface, self.gefs_surface_paths, ts)
        data = gefs_surface_handle.sel(time=ts, channel=self.kept_gefs_surface_channel_names)
        data['x'] = data.lat.values[:,0]
        data['y'] = data.lon.values[0,:]%360
        gefs_surface_field = data.interp(x=lat, y=lon)["values"].values
        if (len(gefs_surface_field.shape) == 4):
            gefs_surface_field = gefs_surface_field[0]
        gefs_surface_field = self.normalize_input(gefs_surface_field, self.means_gefs_surface, self.stds_gefs_surface)

        return gefs_surface_field

    def _get_gefs_isobaric(self, ts, lat, lon):
        '''
        Retrieve GEFS isobaric samples from zarr files
        '''
        gefs_isobaric_handle = self._get_ds_handles(self.ds_gefs_isobaric, self.gefs_isobaric_paths, ts)
        data = gefs_isobaric_handle.sel(time=ts, channel=self.kept_gefs_isobaric_channel_names)
        data['x'] = data.lat.values[:,0]
        data['y'] = data.lon.values[0,:]%360
        gefs_isobaric_field = data.interp(x=lat, y=lon)["values"].values
        if (len(gefs_isobaric_field.shape) == 4):
            gefs_isobaric_field = gefs_isobaric_field[0]
        gefs_isobaric_field = self.normalize_input(gefs_isobaric_field, self.means_gefs_isobaric, self.stds_gefs_isobaric)

        return gefs_isobaric_field

    def normalize_output(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x -= self.means_hrrr
            x /= self.stds_hrrr
        return x

    def denormalize_output(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x *= self.stds_hrrr
            x += self.means_hrrr
        return x

    def _get_hrrr(self, ts, crop_box):
        '''
        Retrieve HRRR samples from zarr files
        '''
        
        hrrr_handle = self._get_ds_handles(self.ds_hrrr, self.hrrr_paths, ts, mask_and_scale=False)
        ds_channel_names = list(np.array(hrrr_handle.channel))
        ((y0, y1), (x0, x1)) = crop_box

        hrrr_field = hrrr_handle.sel(time=ts, channel=self.kept_hrrr_channel_names)["values"][..., y0:y1, x0:x1].values   
        if (len(hrrr_field.shape) == 4):
            hrrr_field = hrrr_field[0]
        hrrr_field = self.normalize_output(hrrr_field)
        return hrrr_field

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        ((y_start, y_end), (x_start, x_end)) = self.hrrr_window
        return (y_end - y_start, x_end - x_start)

    def _get_crop_box(self):
        if self.sample_shape == None:
            return self.hrrr_window

        ((y_start, y_end), (x_start, x_end)) = self.hrrr_window

        y0 = np.random.randint(y_start, y_end - self.sample_shape[0] + 1)
        y1 = y0 + self.sample_shape[0]
        x0 = np.random.randint(x_start, x_end - self.sample_shape[1] + 1)
        x1 = x0 + self.sample_shape[1]
        return ((y0, y1), (x0, x1))

    def __getitem__(self, global_idx):
        '''
        Return data as a dict (so we can potentially add extras, metadata, etc if desired
        '''
        torch.cuda.nvtx.range_push("hrrr_dataloader:get")
        if self.overfit:
            global_idx = 42
        time_index = self._global_idx_to_datetime(global_idx)
        ((y0, y1), (x0, x1)) = crop_box = self._get_crop_box()
        lon = self.hrrr_lon[y0:y1, x0:x1]
        lat = self.hrrr_lat[y0:y1, x0:x1]
        gefs_surface_sample = self._get_gefs_surface(time_index, lon=lon, lat=lat)
        gefs_isobaric_sample = self._get_gefs_isobaric(time_index, lon=lon, lat=lat)
        if self.ds_factor > 1:
            gefs_surface_sample = self._create_lowres_(gefs_surface_sample, factor=self.ds_factor)
            gefs_isobaric_sample = self._create_lowres_(gefs_isobaric_sample, factor=self.ds_factor)
        hrrr_sample = self._get_hrrr(time_index, crop_box=crop_box)
        gefs_sample = np.concatenate((gefs_surface_sample, gefs_isobaric_sample), axis=0)    
        torch.cuda.nvtx.range_pop()
        return hrrr_sample, gefs_sample, global_idx, int(time_index[-2:])//3

    def _global_idx_to_datetime(self, global_idx):
        '''
        Parse a global sample index and return the input/target timstamps as datetimes
        '''
        return self.valid_samples[global_idx]

    def _get_ds_handles(self, handles, paths, ts, mask_and_scale=True):
        '''
        Return handles for the appropriate year
        '''
        if ts[:4]=="2020":
            name = "2020_12_12"
        elif ts[:4]=="2024":
            name = "2024_01_07"
        elif "01"<=ts[4:6]<="06":
            name = ts[:4] + "_01_06"
        elif "07"<=ts[4:6]<="12":
            name = ts[:4] + "_07_12"
        else:
            raise Exception("wrong time")
        
        if name in handles:
            ds_handle = handles[name]
        else:
            ds_handle = xr.open_zarr(paths[name], consolidated=True, mask_and_scale=mask_and_scale)
            handles[name] = ds_handle
        return ds_handle

    @staticmethod
    def _create_lowres_(x, factor=4):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # 8x8x3  #subsample
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x

    def latitude(self):
        return self.hrrr_lat if self.train else self.crop_to_fit(self.hrrr_lat)

    def longitude(self):
        return self.hrrr_lon if self.train else self.crop_to_fit(self.hrrr_lon)

    def time(self):
        return self.valid_samples

    def input_channels(self):
        return [ChannelMetadata(name=n) for n in (self._get_gefs_surface_channel_names() + self._get_gefs_isobaric_channel_names())]

    def output_channels(self):
        return [ChannelMetadata(name=n) for n in self._get_hrrr_channel_names()]

    def info(self):
        return {
            "input_normalization": {
                "gefs_surface": (self.means_gefs_surface.squeeze(), self.stds_gefs_surface.squeeze()),
                "gefs_isobaric": (self.means_gefs_isobaric.squeeze(), self.stds_gefs_isobaric.squeeze())
            },
            "target_normalization": (self.means_hrrr.squeeze(), self.stds_hrrr.squeeze())
        }
