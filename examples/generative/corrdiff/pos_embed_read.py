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

import cftime
from einops import rearrange
import hydra
import netCDF4 as nc
import nvtx
from omegaconf import OmegaConf, DictConfig
import torch
from torch.distributed import gather
import torch._dynamo
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import (
    ablation_sampler,
    parse_int_list,
    StackedRandomGenerator,
)
from modulus import Module
from datasets.base import DownscalingDataset
from datasets.dataset import init_dataset_from_config
from datasets.time import convert_datetime_to_cftime, time_range


time_format = "%Y-%m-%dT%H:%M:%S"
model_type = 'v3'

@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Parse options
    res_ckpt_filename = getattr(cfg, "res_ckpt_filename")
    force_fp16 = getattr(cfg, "force_fp16", False)
    inference_mode = getattr(cfg, "inference_mode", "regression_and_diffusion")

    # Parse data options
    times_range = getattr(cfg, "time_range", None)

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Load diffusion network, move to device, change precision

    net_res = Module.from_checkpoint(res_ckpt_filename)
    net_res = net_res.eval()

    print(net_res.model.pos_embd.shape)

    pos_embd = net_res.model.pos_embd
    pos_embd = torch.nn.functional.avg_pool2d(pos_embd, (4,4), (1,1))
    print(pos_embd.shape)
    input = torch.reshape(pos_embd, (100,-1))
    U, S, Vh = torch.linalg.svd(input, full_matrices=False)
    plt.plot(S.cpu().detach().numpy())
    plt.savefig("pos_embed_svd_value.png")
    Vh = torch.reshape(Vh, (100, 1053, 1789))
    Vh = torch.flip(Vh, (1,))
    print(Vh.shape)

    plt.figure(figsize=(15, 9))
    for i in range(4):
        plt.subplot(2,2,i+1)
        print(torch.max(Vh[i]), torch.min(Vh[i]))
        plt.imshow(Vh[i].cpu().detach().numpy(), vmin=-0.002, vmax=0.002)
    plt.suptitle("SVD components of learnt positional embedding - mvp v3")
    plt.savefig("pos_embed_svd.png")

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
