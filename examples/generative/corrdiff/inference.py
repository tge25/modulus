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
        t_hat = torch.tensor(1.0).to(torch.float64).to(dist.device)

        # Run regression on just a single batch element and then repeat
        output_tensor.append(
            mean_predictor_model(latents, input_tensor, t_hat, lead_time_label=lead_time).to(torch.float64)
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

        class_labels = None
        print(generative_model.label_dim)
        if generative_model.label_dim: # TODO: have no idea what this is
            class_labels = torch.eye(net.label_dim, device=dist.device)[
                rnd.randint(net.label_dim, size=[seed_batch_size], device=device)
            ]

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
            images = sampler_fn(
                net,
                latents,
                img_lr,
                class_labels,
                randn_like=torch.randn_like,
                **sampler_kwargs,
            )
        output_tensor.append(images)

    # Return output tensor
    return torch.cat(output_tensor)




#def generate(
#    net,
#    seeds,
#    seed_batch_size,
#    img_shape,  # as (img_shape_x, img_shape_y)
#    img_out_channels,
#    sampling_method=None,
#    img_lr=None,
#    pretext=None,
#    **sampler_kwargs,
#):
#    """Generate random images using the techniques described in the paper
#    "Elucidating the Design Space of Diffusion-Based Generative Models".
#    """
#
#    if sampling_method == "stochastic":
#        # import stochastic sampler
#        try:
#            from edmss import edm_sampler
#        except ImportError:
#            raise ImportError(
#                "Please get the edm_sampler by running: pip install git+https://github.com/mnabian/edmss.git"
#            )
#        sampler_kwargs.update(({"img_shape": img_shape}))
#
#    # Instantiate distributed manager.
#    dist = DistributedManager()
#    device = dist.device
#
#    num_batches = (
#        (len(seeds) - 1) // (seed_batch_size * dist.world_size) + 1
#    ) * dist.world_size
#    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
#    rank_batches = all_batches[dist.rank :: dist.world_size]
#
#    # Synchronize
#    if dist.world_size > 1:
#        torch.distributed.barrier()
#
#    img_lr = img_lr.to(memory_format=torch.channels_last)
#
#    # Loop over batches.
#    all_images = []
#    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.rank != 0)):
#        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
#            batch_size = len(batch_seeds)
#            if batch_size == 0:
#                continue
#
#            # Pick latents and labels.
#            rnd = StackedRandomGenerator(device, batch_seeds)
#
#            if model_type == 'v3':
#                latents = rnd.randn(
#                    [
#                        seed_batch_size,
#                        img_out_channels + 1,
#                        img_shape[1],
#                        img_shape[0],
#                    ],
#                    device=device,
#                ).to(memory_format=torch.channels_last)
#
#            else:
#                latents = rnd.randn(
#                    [
#                        seed_batch_size,
#                        img_out_channels,
#                        img_shape[1],
#                        img_shape[0],
#                    ],
#                    device=device,
#                ).to(memory_format=torch.channels_last)
#
#            class_labels = None
#            if net.label_dim:
#                class_labels = torch.eye(net.label_dim, device=device)[
#                    rnd.randint(net.label_dim, size=[seed_batch_size], device=device)
#                ]
#
#            # Generate images.
#            sampler_kwargs = {
#                key: value for key, value in sampler_kwargs.items() if value is not None
#            }
#            if pretext == "gen":
#                if sampling_method == "deterministic":
#                    sampler_fn = ablation_sampler
#                elif sampling_method == "stochastic":
#                    sampler_fn = edm_sampler
#                else:
#                    raise ValueError(
#                        f"Unknown sampling method {sampling_method}. Should be either 'stochastic' or 'deterministic'."
#                    )
#            elif pretext == "reg":
#                latents = torch.zeros_like(latents, memory_format=torch.channels_last)
#                sampler_fn = unet_regression
#            else:
#                raise ValueError(
#                    f"Unknown pretext {pretext}. Should be either 'gen' or 'reg'."
#                )
#
#            with torch.inference_mode():
#                images = sampler_fn(
#                    net,
#                    latents,
#                    img_lr,
#                    class_labels,
#                    randn_like=torch.randn_like,
#                    **sampler_kwargs,
#                )
#            all_images.append(images)
#    return torch.cat(all_images)


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

    # Run inference on generative model TODO: Implement
    output_gen = _generative_model_inference(
        input_tensor=input_tensor,
        lead_time=lead_time,
        generative_model=generative_model,
        dist=dist,
        output_channels=output_channels,
        rank_seeds=rank_seeds,
        sampling_kwargs=sampling_kwargs,
    )

    # Combine regression and residual images
    output = output_mean + output_gen

    # Return output
    return output

if __name__ == "__main__":

    # Parameters
    output_channels = 3
    shape_x = 1056
    shape_y = 1792
    sigma_min = None
    sigma_max = None
    rho = 7
    S_churn = 0.0
    S_min = 0.0
    S_max = 0.0
    S_noise = 0.0
    sampler_kwargs = {
        "patch_shape": 448,
        "overlap_pix": 4,
        "boundary_pix": 2,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": rho,
        "S_churn": S_churn,
        "S_min": S_min,
        "S_max": S_max,
        "S_noise": S_noise,
    }
 
    # Get distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Load model
    mean_predictor_model_path = "./UNet.0.1960960.mdlus"
    mean_predictor_model = Module.from_checkpoint(mean_predictor_model_path)
    mean_predictor_model = mean_predictor_model.to(dist.device)
    generative_model_path = "./EDMPrecondSRV2.0.5821440.mdlus"
    generative_model = Module.from_checkpoint(generative_model_path)
    generative_model = generative_model.to(dist.device)

    # Call inference function 
    input_tensor = torch.zeros((1, 42, shape_x, shape_y)).to(dist.device)
    output_tensor = inference(
        input_tensor=input_tensor,
        lead_time=torch.Tensor([1]).to(dist.device).to(torch.int32),
        generative_model=generative_model,
        mean_predictor_model=mean_predictor_model,
        seeds=[0],
        dist=dist,
        output_channels=output_channels,
        sampling_kwargs=sampler_kwargs
    )
    print(output_tensor)
