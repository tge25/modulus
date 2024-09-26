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

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from concurrent.futures import ThreadPoolExecutor
import datetime
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from modulus.distributed import DistributedManager
from datasets.dataset import init_dataset_from_config
from datasets.time import convert_datetime_to_cftime, time_range
import numpy as np

time_format = "%Y-%m-%dT%H:%M:%S"
model_type = 'v3'

@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    # Parse data options
    times_range = getattr(cfg, "time_range", None)

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    if times_range is not None:
        times = []
        t_start = datetime.datetime.strptime(times_range[0], time_format)
        t_end = datetime.datetime.strptime(times_range[1], time_format)
        dt = datetime.timedelta(hours=(times_range[2] if len(times_range) > 2 else 1))
        times = [
            t.strftime(time_format)
            for t in time_range(t_start, t_end, dt, inclusive=True)
        ]
    else:
        times = getattr(cfg, "times", ["2021-02-02T00:00:00"])

    # Create dataset object
    print(times)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, sampler = get_dataset_and_sampler(dataset_cfg=dataset_cfg, times=times)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
    )
    lat=dataset.latitude().to_numpy()
    lon=dataset.longitude().to_numpy()
    times = dataset.time()
    print(times)
    for image_tar, image_lr, index, lead_time in iter(data_loader):
        image_tar = dataset.denormalize_output(image_tar.numpy())
        plt.figure(figsize=(18,5))
        plt.subplot(1,3,1)
        print(lat.shape, np.reshape(image_tar[0,0], -1).shape)
        plt.pcolormesh(lon, lat, image_tar[0,0], vmin=-15, vmax=28)
        plt.colorbar()
        plt.title("u10m")
        plt.subplot(1,3,2)
        plt.pcolormesh(lon, lat, image_tar[0,1], vmin=-22, vmax=22)
        plt.colorbar()
        plt.title("v10m")
        plt.subplot(1,3,3)
        plt.pcolormesh(lon, lat, image_tar[0,2], vmin=245, vmax=315)
        plt.colorbar()
        plt.title("t2m")
        plt.tight_layout()
        plt.savefig(f"./output_movie/{lead_time[0]}.png")
    import imageio 
    import os
    writer = imageio.get_writer(f'20240227.mp4', fps = 3)
    dirFiles = os.listdir('./output_movie/') # list of directory files
    dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for im in dirFiles:
        writer.append_data(imageio.imread('./output_movie/'+im))
    writer.close()
    print("saved to twc_mvp1_movie.mp4")


def _get_name(channel_info):
    return channel_info.name + channel_info.level


def get_dataset_and_sampler(dataset_cfg, times):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    plot_times = [time for time in times]
    all_times = dataset.time()
    time_indices = [all_times.index(t) for t in plot_times]
    sampler = time_indices

    return dataset, sampler


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
