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
from modulus.utils.generative import (
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
    lead_time: torch.Tensor,
    mean_predictor_model: Module,
    output_channels: int = 8,
):
    """ Run inference on the mean predictor model.

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input tensor to run inference on. shape: (1, C, H, W), eg (1, 37, 1056, 1792)
    lead_time: torch.Tensor
        Lead time tensor. shape: (1)
    mean_predictor_model: Module
        Mean predictor model to run inference on.
    output_channels: int
        Number of output channels to generate. eg 8.

    Returns
    -------
    output_tensor: torch.Tensor
        Output tensor from the mean predictor model. shape: (N, C, H, W), eg (1, 8, 1056, 1792)
    """

    # Generate latents (TODO: V3 Specific)
    latents = torch.zeros(
        (
            1,
            output_channels + 1, # TODO: Hack for V3, remove +1
            input_tensor.shape[2],
            input_tensor.shape[3],
        ),
        device=input_tensor.device,
    ).to(memory_format=torch.channels_last)

    # Main sampling loop.
    t_hat = torch.tensor(1.0).to(torch.float32).to(input_tensor.device)

    # Run regression on just a single batch element and then repeat
    with torch.inference_mode():
        output_tensor = (
            mean_predictor_model(latents, input_tensor, t_hat, lead_time_label=lead_time).to(torch.float32)
        )
    return output_tensor


def _generative_model_inference(
    input_tensor: torch.Tensor,
    lead_time: int,
    mean_hr: torch.Tensor,
    generative_model: Module,
    output_channels: int = 8,
    seeds: list = [0],
    sampling_kwargs: dict = {},
):
    """ Run inference on the generative model.

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input tensor to run inference on. shape: (1, C, H, W), eg (1, 37, 1056, 1792)
    lead_time: torch.Tensor
        Lead time tensor. shape: (1)
    mean_hr: torch.Tensor
        Mean predictor output tensor. shape: (1, C, H, W), eg (1, 8, 1056, 1792)
    generative_model: Module
        Generative model to run inference on.
    output_channels: int
        Number of output channels to generate. eg 8.
    seeds: list
        List of seeds to use for inference. Each seed will generate a different output.
    sampling_kwargs: dict
        Dictionary containing sampling parameters for the generative model.

    Returns
    -------
    output_tensor: torch.Tensor
        Output tensor from the generative model. shape: (len(seeds), C, H, W), eg (1, 8, 1056, 1792)
    """

    # Check if sampling kwargs has required keys
    required_keys = {
        "img_shape",
        "patch_shape",
        "overlap_pix",
        "boundary_pix",
        "num_steps",
        "rho",
        "S_churn",
        "S_min",
        "S_max",
        "S_noise",
    }
    assert required_keys.issubset(sampling_kwargs.keys()), f"Missing keys in sampling_kwargs: {required_keys - set(sampling_kwargs.keys())}"

    # Loop over seeds in batch
    output_tensor = []
    for seed in seeds:

        # Instantiate random generator
        rnd = StackedRandomGenerator(
            input_tensor.device,
            [seed],
        )

        # Generate latents (TODO: V3 Specific)
        latents = rnd.randn(
            [
                1,
                output_channels + 1, # TODO: Hack for V3, remove +1 if possible
                input_tensor.shape[2],
                input_tensor.shape[3],
            ],
            device=input_tensor.device,
        ).to(memory_format=torch.channels_last)

        # Run inference
        with torch.inference_mode():
            images = edm_sampler(
                generative_model,
                latents,
                input_tensor,
                class_labels=None,
                randn_like=torch.randn_like,
                mean_hr=mean_hr,
                lead_time_label=lead_time,
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
    output_channels: int = 8,
    sampling_kwargs: dict = {},
):
    """Function to generate an image CorrDiff US model.

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input tensor to run inference on. shape: (1, C, H, W), eg (1, 37, 1056, 1792)
    lead_time: torch.Tensor
        Lead time tensor. shape: (1)
    generative_model: Module
        Generative model to run inference on.
    mean_predictor_model: Module
        Mean predictor model to run inference on.
    seeds: list
        List of seeds to use for inference. Each seed will generate a different output.
    output_channels: int
        Number of output channels to generate. eg 8.
    sampling_kwargs: dict
        Dictionary containing sampling parameters for the generative model.

    Returns
    -------
    output_tensor: torch.Tensor
        Output tensor from the generative model. shape: (len(seeds), C, H, W), eg (1, 8, 1056, 1792)
    """

    # Memory format
    input_tensor = input_tensor.to(memory_format=torch.channels_last)

    # Run inference on mean predictor
    output_mean = _mean_predictor_inference(
        input_tensor=input_tensor,
        lead_time=lead_time,
        mean_predictor_model=mean_predictor_model,
        output_channels=output_channels,
    )

    # Run inference on generative model
    output_gen = _generative_model_inference(
        input_tensor=input_tensor,
        lead_time=lead_time,
        mean_hr=output_mean,
        generative_model=generative_model,
        output_channels=output_channels,
        seeds=seeds,
        sampling_kwargs=sampling_kwargs,
    )

    # Combine regression and residual images
    output = output_mean + output_gen # TODO: check reshaping here

    # Return output
    return output

if __name__ == "__main__":

    # Parameters
    output_channels = 8 # Maybe 9 for V3
    shape_x = 1056
    shape_y = 1792
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
 
    # Get Data
    data_path = "/home/oliver/unleashed/validation_report/twc_mvp_v3_full1_0.nc"
    ds = xr.open_dataset(data_path, group="input").isel(time=0)
    input_tensor = torch.zeros((1, 37, shape_x, shape_y))
    for i, var in enumerate(ds.data_vars):
        input_tensor[0, i] = torch.tensor(ds[var].values)
    del ds

    # Load mean predictor model
    mean_predictor_model_path = "./UNet.0.1960960.mdlus"
    mean_predictor_model = Module.from_checkpoint(mean_predictor_model_path)
    mean_predictor_model = mean_predictor_model
    mean_predictor_model.eval().to(memory_format=torch.channels_last)
    mean_predictor_model.use_fp16 = True

    # Load generative model
    generative_model_path = "./EDMPrecondSRV2.0.5821440.mdlus"
    generative_model = Module.from_checkpoint(generative_model_path)
    generative_model = generative_model
    generative_model.eval().to(memory_format=torch.channels_last)
    generative_model.use_fp16 = True

    # Call inference function 
    output_tensor = inference(
        input_tensor=input_tensor,
        lead_time=torch.Tensor([1]).to(torch.float32),
        generative_model=generative_model,
        mean_predictor_model=mean_predictor_model,
        seeds=[0],
        output_channels=output_channels,
        sampling_kwargs=sampler_kwargs
    )

    # Save output
    import matplotlib.pyplot as plt
    for i in range(output_tensor.shape[1]):
        plt.imshow(output_tensor[0, i].cpu().numpy())
        plt.savefig(f"output_{i}.png")