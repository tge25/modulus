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

#try:
#    from edmss import edm_sampler
#except ImportError:
#    raise ImportError(
#        "Please get the edm_sampler by running: pip install git+https://github.com/mnabian/edmss.git"
#    )
 

def _mean_predictor_inference(
    input_tensor: torch.Tensor,
    lead_time: int,
    mean_predictor_model: Module,
    dist: DistributedManager, # (TODO: Maybe refactor to remove this?)
    output_channels: int = 3,
    sigma_min: float = 0.0,
    sigma_max: float = 0.0,
    num_steps: int = 2,
    rho: int = 7,
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
                input_tensor.shape[1],
                input_tensor.shape[2],
            ),
            device=dist.device,
        ).to(memory_format=torch.channels_last)

        # Adjust noise levels based on what's supported by the network. (TODO: Probably not needed)
        sigma_min = max(sigma_min, mean_predictor_model.sigma_min)
        sigma_max = min(sigma_max, mean_predictor_model.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [mean_predictor_model.round_sigma(t_steps), torch.zeros_like(t_steps[:1]).to(dist.device)]
        )

        # Main sampling loop.
        x_hat = latents.to(torch.float64) * t_steps[0] # TODO: Ask Tao, this will always be zero
        t_hat = torch.tensor(1.0).to(torch.float64).to(dist.device)

        # Run regression on just a single batch element and then repeat
        output_tensor.append(
            mean_predictor_model(x_hat[0:1], input_tensor, t_hat, lead_time_label=lead_time).to(torch.float64)
        )

    # Return output tensor
    return torch.cat(output_tensor)


def _generative_model_inference(
    input_tensor: torch.Tensor,
    lead_time: int,
    mean_predictor_model: Module,
    seeds: list,
    dist: DistributedManager, # (TODO: Maybe refactor to remove this?)
    sampling_method: str = "stochastic",
):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    pass

    ## Get seeds for current rank
    #rank_seeds = torch.as_tensor(seeds)[dist.rank :: dist.world_size]

    ## Synchronize TODO: Remove this?
    #if dist.world_size > 1:
    #    torch.distributed.barrier()

    ## Loop over seeds in batch
    #all_images = []
    #for seed in rank_seeds:

    #    # Instantiate random generator
    #    rnd = StackedRandomGenerator(
    #        dist.device,
    #        [seed],
    #    )

    #    # Generate latents (TODO: V3 Specific)
    #    latents = rnd.randn(
    #        [
    #            1,
    #            img_out_channels + 1, # TODO: Hack for V3, remove +1
    #            input_tensor.shape[1],
    #            input_tensor.shape[2],
    #        ],
    #        device=dist.device,
    #    ).to(memory_format=torch.channels_last)

    #    class_labels = None
    #    if net.label_dim: # TODO: have no idea what this is
    #        class_labels = torch.eye(net.label_dim, device=dist.device)[
    #            rnd.randint(net.label_dim, size=[seed_batch_size], device=device)
    #        ]

    #    # Generate latent images
    #    latents = torch.zeros_like(latents, memory_format=torch.channels_last)

    #    # Adjust noise levels based on what's supported by the network.
    #    sigma_min = max(sigma_min, net.sigma_min)
    #    sigma_max = min(sigma_max, net.sigma_max)

    #    # Time step discretization.
    #    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    #    t_steps = (
    #        sigma_max ** (1 / rho)
    #        + step_indices
    #        / (num_steps - 1)
    #        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    #    ) ** rho
    #    t_steps = torch.cat(
    #        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    #    )  # t_N = 0

    #    x_lr = img_lr

    #    # Main sampling loop.
    #    x_hat = latents.to(torch.float64) * t_steps[0]
    #    t_hat = torch.tensor(1.0).to(torch.float64).cuda()

    #    # Run regression on just a single batch element and then repeat
    #    x_next = net(x_hat[0:1], x_lr, t_hat, class_labels, lead_time_label=kwargs["lead_time_label"]).to(torch.float64)

    #    if x_hat.shape[0] > 1:
    #        x_next = x_next.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])

    #    return x_next



    #    with torch.inference_mode():
    #        images = sampler_fn(
    #            net,
    #            latents,
    #            img_lr,
    #            class_labels,
    #            randn_like=torch.randn_like,
    #            **sampler_kwargs,
    #        )
    #    all_images.append(images)
    #return torch.cat(all_images)

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
):
    """Function to generate an image
    """

    # Memory format
    input_tensor = input_tensor.to(memory_format=torch.channels_last)

    # Run inference on mean predictor
    output_mean = _mean_predictor_inference(
        input_tensor=input_tensor,
        lead_time=lead_time,
        mean_predictor_model=mean_predictor_model,
        dist=dist,
        sigma_min=0.0,
        sigma_max=0.0,
        num_steps=2,
        rho=7,
    )

    # Run inference on generative model TODO: Implement
    output_gen = torch.zeros_like(input_tensor)

    # Combine regression and residual images
    output = output_mean + output_gen

    # Return output
    return output

if __name__ == "__main__":

    # Parameters
    patch_shape_x = 448
    patch_shape_y = 448
  
    # Get distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Load model
    mean_predictor_model_path = "./UNet.0.1960960.mdlus"
    mean_predictor_model = Module.from_checkpoint(mean_predictor_model_path)

    # Call inference function 
    input_tensor = torch.zeros((1, 3, patch_shape_x, patch_shape_y))
    output_tensor = inference(
        input_tensor=input_tensor,
        lead_time=1,
        generative_model=None,
        mean_predictor_model=mean_predictor_model,
        seeds=[0],
        dist=dist,
    )
    print(output_tensor)




