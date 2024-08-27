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

""" Simple inference function for CorrDiff US model.
"""

import torch
import math
import xarray as xr

from modulus import Module
from modulus.distributed import DistributedManager
from modulus.utils.generative import (
    ablation_sampler,
    StackedRandomGenerator,
)

try:
    from edmss import edm_sampler
except ImportError:
    raise ImportError(
        "Please get the edm_sampler by running: pip install git+https://github.com/mnabian/edmss.git"
    )
 

def _mean_predictor_inference(
    input_tensor: torch.Tensor,
    lead_time: int,
    mean_predictor_model: Module,
    dist: DistributedManager, # (TODO: Maybe refactor to remove this?)
    output_channels: int = 3,
    rank_seeds: list = [0],
):
    """ Run inference on the mean predictor model.
    """

    # Loop over seeds in batch
    output_tensor = []
    for seed in rank_seeds:

        # Generate latents (TODO: V3 Specific)
        latents = torch.zeros(
            (
                1,
                output_channels + 1, # TODO: Hack for V3, remove +1
                input_tensor.shape[2],
                input_tensor.shape[3],
            ),
            device=dist.device,
        ).to(memory_format=torch.channels_last)

        # Main sampling loop.
        t_hat = torch.tensor(1.0).to(torch.float32).to(dist.device)

        # Run regression on just a single batch element and then repeat
        with torch.inference_mode():
            output_tensor.append(
                mean_predictor_model(latents, input_tensor, t_hat, lead_time_label=lead_time).to(torch.float32)
            )

    # Return output tensor
    return torch.cat(output_tensor)


def _generative_model_inference(
    input_tensor: torch.Tensor,
    lead_time: int,
    generative_model: Module,
    dist: DistributedManager, # (TODO: Maybe refactor to remove this?)
    output_channels: int = 3,
    rank_seeds: list = [0],
    sampling_kwargs: dict = {},
    sampling_method: str = "stochastic",
):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Loop over seeds in batch
    output_tensor = []
    for seed in rank_seeds:

        # Instantiate random generator
        rnd = StackedRandomGenerator(
            dist.device,
            [seed],
        )

        # Generate latents (TODO: V3 Specific)
        latents = rnd.randn(
            [
                1,
                output_channels + 1, # TODO: Hack for V3, remove +1
                input_tensor.shape[2],
                input_tensor.shape[3],
            ],
            device=dist.device,
        ).to(memory_format=torch.channels_last)

        # Sampling method
        sampler_kwargs = {
            key: value for key, value in sampling_kwargs.items() if value is not None
        }
        if sampling_method == "deterministic":
            sampler_fn = ablation_sampler
        elif sampling_method == "stochastic":
            sampler_fn = edm_sampler
        else:
            raise ValueError(
                f"Unknown sampling method {sampling_method}. Should be either 'stochastic' or 'deterministic'."
            )

        # Run inference
        with torch.inference_mode():
            #print(latents.dtype)
            #print(input_tensor.dtype)
            #print(sampler_kwargs["mean_hr"].dtype)
            images = sampler_fn(
                generative_model,
                latents,
                input_tensor,
                class_labels=None,
                randn_like=torch.randn_like,
                **sampler_kwargs,
            )
        output_tensor.append(images)

    # Return output tensor
    return torch.cat(output_tensor)


def inference(
    input_tensor: torch.Tensor,
    lead_time: int,
    generative_model: Module,
    mean_predictor_model: Module,
    seeds: list,
    dist: DistributedManager,
    output_channels: int = 3,
    sampling_kwargs: dict = {},
):
    """Function to generate an image
    """

    # Memory format
    input_tensor = input_tensor.to(memory_format=torch.channels_last)

    # Get rank seeds
    rank_seeds = torch.as_tensor(seeds)[dist.rank :: dist.world_size]

    # Run inference on mean predictor
    output_mean = _mean_predictor_inference(
        input_tensor=input_tensor,
        lead_time=lead_time,
        mean_predictor_model=mean_predictor_model,
        dist=dist,
        output_channels=output_channels,
        rank_seeds=rank_seeds,
    )

    # TODO: Hacky
    sampling_kwargs["mean_hr"] = output_mean
    sampling_kwargs["lead_time_label"] = lead_time
    #print(output_mean.shape)

    ## Run inference on generative model TODO: Implement
    #output_gen = _generative_model_inference(
    #    input_tensor=input_tensor,
    #    lead_time=lead_time,
    #    generative_model=generative_model,
    #    dist=dist,
    #    output_channels=output_channels,
    #    rank_seeds=rank_seeds,
    #    sampling_kwargs=sampling_kwargs,
    #)

    ## Combine regression and residual images
    #output = output_mean + output_gen
    output = output_mean
    print(output.shape)

    # Return output
    return output

if __name__ == "__main__":

    # Parameters
    output_channels = 8 # Maybe 9 for V3
    shape_x = 1056
    shape_y = 1792
    img_shape = (shape_y, shape_x)
    sampler_kwargs = {
        "num_steps": 18,
        "patch_shape": 448,
        "overlap_pix": 4,
        "boundary_pix": 2,
        "rho": 7,
        "S_churn": 0,
        "S_min": 0,
        "S_max": math.inf,
        "S_noise": 1,
        "img_shape": (shape_y, shape_x),
    }
 
    # Get distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Get Data
    data_path = "/home/oliver/unleashed/validation_report/twc_mvp_v3_full1_0.nc"
    ds = xr.open_dataset(data_path, group="input").isel(time=0)
    input_tensor = torch.zeros((1, 37, shape_x, shape_y)).to(dist.device)
    for i, var in enumerate(ds.data_vars):
        input_tensor[0, i] = torch.tensor(ds[var].values).to(dist.device)
    del ds

    # Load mean predictor model
    mean_predictor_model_path = "./UNet.0.1960960.mdlus"
    mean_predictor_model = Module.from_checkpoint(mean_predictor_model_path)
    mean_predictor_model = mean_predictor_model.to(dist.device)
    mean_predictor_model.eval().to(memory_format=torch.channels_last)
    mean_predictor_model.use_fp16 = True

    # Load generative model
    generative_model_path = "./EDMPrecondSRV2.0.5821440.mdlus"
    generative_model = Module.from_checkpoint(generative_model_path)
    generative_model = generative_model.to(dist.device)
    generative_model.eval().to(memory_format=torch.channels_last)
    generative_model.use_fp16 = True

    # Call inference function 
    output_tensor = inference(
        input_tensor=input_tensor,
        lead_time=torch.Tensor([1]).to(dist.device).to(torch.float32),
        generative_model=generative_model,
        mean_predictor_model=mean_predictor_model,
        seeds=[0],
        dist=dist,
        output_channels=output_channels,
        sampling_kwargs=sampler_kwargs
    )
    import matplotlib.pyplot as plt
    plt.imshow(output_tensor[0, 0].cpu().numpy())
    plt.show()
