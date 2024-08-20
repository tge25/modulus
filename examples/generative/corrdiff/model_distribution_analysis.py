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

import os, time, psutil, hydra, torch
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from modulus import Module
from modulus.models.diffusion import UNet, EDMPrecondSRV2
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.metrics.diffusion import RegressionLoss, ResLoss, RegressionLossEntropy, ResLossEntropy, ResLoss5Types
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
from datasets.dataset import init_train_valid_datasets_from_config
import random
import time
import modulus
from pathlib import Path
from typing import Dict, List, NewType, Union
from train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    parse_model_args,
)

class ResOut:
    """
    Mixture loss function for denoising score matching.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net,
        img_shape_x,
        img_shape_y,
        patch_shape_x,
        patch_shape_y,
        patch_num,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        self.unet = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.patch_shape_x = patch_shape_x
        self.patch_shape_y = patch_shape_y
        self.patch_num = patch_num
        self.hr_mean_conditioning = hr_mean_conditioning
        self.entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, net, img_clean, img_lr, lead_time_label=None, labels=None, augment_pipe=None):
        """
        Calculate and return the loss for denoising score matching.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        img_clean: torch.Tensor
            Input images (high resolution) to the neural network.

        img_lr: torch.Tensor
            Input images (low resolution) to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        precip = torch.sum(img_clean[:,4:, ], axis=1, keepdim = True)
        hrrr_non_precip = ((precip == 0).float())
        img_clean = torch.cat((img_clean, hrrr_non_precip), dim=1)
        img_clean[:, 4:] = img_clean[:, 4:]/torch.sum(img_clean[:,4:, ], axis=1, keepdim = True)
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr

        # global index
        b = y.shape[0]
        Nx = torch.arange(self.img_shape_x).int()
        Ny = torch.arange(self.img_shape_y).int()
        grid = torch.stack(torch.meshgrid(Ny, Nx, indexing="ij"), dim=0)[
            None,
        ].expand(b, -1, -1, -1)

        # form residual
        y_mean = self.unet(
            torch.zeros_like(y, device=img_clean.device),
            y_lr_res,
            sigma,
            labels,
            lead_time_label=lead_time_label,
            augment_labels=augment_labels,
        )

        y[:,:4] = y[:,:4] - y_mean[:,:4]

        if self.hr_mean_conditioning:
            y_lr = torch.cat((y_mean, y_lr), dim=1).contiguous()
        global_index = None

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        if (
            self.img_shape_x != self.patch_shape_x
            or self.img_shape_y != self.patch_shape_y
        ):  
            
            c_in = y_lr.shape[1]
            c_out = y.shape[1]
            sigma = torch.ones(
                [img_clean.shape[0] * self.patch_num, 1, 1, 1], device=img_clean.device
            )
            weight = (sigma**2 + self.sigma_data**2) / (
                sigma * self.sigma_data
            ) ** 2

            # global interpolation
            input_interp = torch.nn.functional.interpolate(
                img_lr,
                (self.patch_shape_y, self.patch_shape_x),
                mode="bilinear",
            )

            # patch generation from a single sample (not from random samples due to memory consumption of regression)
            y_new = torch.zeros(
                b * self.patch_num,
                c_out,
                self.patch_shape_y,
                self.patch_shape_x,
                device=img_clean.device,
            )
            y_lr_new = torch.zeros(
                b * self.patch_num,
                c_in + input_interp.shape[1],
                self.patch_shape_y,
                self.patch_shape_x,
                device=img_clean.device,
            )
            global_index = torch.zeros(
                b * self.patch_num,
                2,
                self.patch_shape_y,
                self.patch_shape_x,
                dtype=torch.int,
                device=img_clean.device,
            )
            for i in range(self.patch_num):
                rnd_x = random.randint(0, self.img_shape_x - self.patch_shape_x)
                rnd_y = random.randint(0, self.img_shape_y - self.patch_shape_y)
                y_new[b * i : b * (i + 1),] = y[
                    :,
                    :,
                    rnd_y : rnd_y + self.patch_shape_y,
                    rnd_x : rnd_x + self.patch_shape_x,
                ]
                global_index[b * i : b * (i + 1),] = grid[
                    :,
                    :,
                    rnd_y : rnd_y + self.patch_shape_y,
                    rnd_x : rnd_x + self.patch_shape_x,
                ]
                y_lr_new[b * i : b * (i + 1),] = torch.cat(
                    (
                        y_lr[
                            :,
                            :,
                            rnd_y : rnd_y + self.patch_shape_y,
                            rnd_x : rnd_x + self.patch_shape_x,
                        ],
                        input_interp,
                    ),
                    1,
                )
            y = y_new
            y_lr = y_lr_new
        latent = y + torch.randn_like(y) * sigma

        D_yn = net(
            latent,
            y_lr,
            sigma,
            labels,
            global_index=global_index,
            lead_time_label=lead_time_label, 
            augment_labels=augment_labels,
        )

        return D_yn, y

# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()
    torch.autograd.set_detect_anomaly(True)
    # Initialize loggers
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    # Resolve and parse configs
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)  # TODO needs better handling
    if hasattr(cfg, "validation_dataset"):
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation_dataset)
    else:
        validation_dataset_cfg = None
    fp_optimizations = cfg.training.perf.fp_optimizations
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    logger.info(f"Saving the outputs in {os.getcwd()}")

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2,
    }
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
    )

    # Parse image configuration & update model args
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    reg_img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels()) + 1
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # Parse the patch shape
    if 'training' in cfg and 'hp' in cfg['training'] and 'patch_shape_x' in cfg['training']['hp']:
        patch_shape_x = cfg.training.hp.patch_shape_x
    else:
        patch_shape_x = None
    if 'training' in cfg and 'hp' in cfg['training'] and 'patch_shape_y' in cfg['training']['hp']:
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_y = None
    patch_shape = (patch_shape_y, patch_shape_x)    
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)  
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")
    # interpolate global channel if patch-based model is used
    if img_shape[1] != patch_shape[1]:
        img_in_channels += dataset_channels

    # Instantiate the model and move to device.
    
    # Load the regression checkpoint if applicable
    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected a this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        regression_net = Module.from_checkpoint(regression_checkpoint_path)
        #regression_net = Module.from_checkpoint(regression_checkpoint_path)
        regression_net.eval().requires_grad_(False).to(dist.device)
        logger0.success("Loaded the pre-trained regression model")

    # Instantiate the loss function
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    if cfg.model.name in ("diffusion", "patched_diffusion"):
        loss_fn = ResOut(
            regression_net=regression_net,
            img_shape_x=img_shape[1],
            img_shape_y=img_shape[0],
            patch_shape_x=patch_shape[1],
            patch_shape_y=patch_shape[0],
            P_mean=getattr(cfg.training.hp, "P_mean", 0),
            patch_num=patch_num,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )

    paths = ["/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.5120.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.107520.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.158720.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.209920.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.517120.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.1034240.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.1546240.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.2012160.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.3041280.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.4019200.mdlus",
             "/lustre/fsw/coreai_climate_earth2/corrdiff/training_output/diffusion_patch_twc_mvp1_2/checkpoints/EDMPrecondSRV2.0.5048320.mdlus",]

    names = ["5120",
             "107520",
             "158720",
             "209920",
             "517120",
             "1034240",
             "1546240",
             "2012160",
             "3041280",
             "4019200",
             "5048320",]
    
    for i, path in enumerate(paths):
        model = Module.from_checkpoint(path)
        model.eval().requires_grad_(False).to(dist.device)

        ## Resume training from previous checkpoints if exists
        if dist.world_size > 1:
            torch.distributed.barrier()

        ############################################################################
        #                            MAIN TRAINING LOOP                            #
        ############################################################################
        conv2 = None
        lt_embd2 = None
        logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
        cur_tick = 0
        tick_start_time = time.time()
        # Validation
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            with torch.no_grad():
                in_gpu_pred = []
                in_gpu_target = []
                for _ in range(50):
                    img_clean_valid, img_lr_valid, labels_valid, lead_time_label_valid = next(
                        validation_dataset_iterator
                    )

                    lead_time_label_valid = lead_time_label_valid.to(dist.device).contiguous()

                    img_clean_valid = (
                        img_clean_valid.to(dist.device)
                        .to(torch.float32)
                        .contiguous()
                    )
                    img_lr_valid = (
                        img_lr_valid.to(dist.device).to(torch.float32).contiguous()
                    )
                    labels_valid = labels_valid.to(dist.device).contiguous()
                    pred, target = loss_fn(
                        net=model,
                        img_clean=img_clean_valid,
                        img_lr=img_lr_valid,
                        labels=labels_valid,
                        lead_time_label=lead_time_label_valid,
                        augment_pipe=None,
                    )
                    pred[:,4:] = pred[:,4:].softmax(dim=1)
                    in_gpu_pred.append(pred)
                    in_gpu_target.append(target)

                in_gpu_pred = torch.cat(in_gpu_pred, dim=0)
                in_gpu_target = torch.cat(in_gpu_target, dim=0)
                if dist.rank == 0:
                    print(i, in_gpu_pred.shape, time.time()-tick_start_time, flush=True)    
                print("torch", torch.max(in_gpu_pred[:,4:]), torch.min(in_gpu_pred[:,4:]), flush=True)
                in_gpu_pred = in_gpu_pred.cpu().detach().numpy()
                in_gpu_target = in_gpu_target.cpu().detach().numpy()    
                print("numpy", np.max(in_gpu_pred[:,4:]), np.min(in_gpu_pred[:,4:]), flush=True)       
                np.savez("/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/test/"+names[i]+f"_{dist.rank}", pred = in_gpu_pred, target = in_gpu_target)
    # Done.
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
