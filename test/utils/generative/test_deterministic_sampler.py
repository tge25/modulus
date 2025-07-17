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


import pytest
import torch
from pytest_utils import import_or_fail


# Mock a minimal net class for testing
class MockNet:
    def __init__(self, sigma_min=0.0, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x, img_lr, sigma, class_labels):
        return x  # Mock behavior of net

    def round_sigma(self, sigma):
        return torch.tensor(sigma)


# Define a fixture for the network
@pytest.fixture
def mock_net():
    return MockNet()


# Basic functionality test
@import_or_fail("cftime")
def test_deterministic_sampler_output_type_and_shape(mock_net, pytestconfig):

    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(net=mock_net, latents=latents, img_lr=img_lr)
    assert isinstance(output, torch.Tensor)
    assert output.shape == latents.shape


# Test for parameter validation
@import_or_fail("cftime")
@pytest.mark.parametrize("solver", ["invalid_solver", "euler", "heun"])
def test_deterministic_sampler_solver_validation(mock_net, solver, pytestconfig):

    from physicsnemo.utils.diffusion import deterministic_sampler

    if solver == "invalid_solver":
        with pytest.raises(ValueError):
            deterministic_sampler(
                net=mock_net,
                latents=torch.randn(1, 3, 64, 64),
                img_lr=torch.randn(1, 3, 64, 64),
                solver=solver,
            )
    else:
        # No exception should be raised for valid solvers
        deterministic_sampler(
            net=mock_net,
            latents=torch.randn(1, 3, 64, 64),
            img_lr=torch.randn(1, 3, 64, 64),
            solver=solver,
        )


# Test for edge cases
@import_or_fail("cftime")
def test_deterministic_sampler_edge_cases(mock_net, pytestconfig):

    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    # Test with extreme rho values, zero noise levels, etc.
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, rho=1000, sigma_min=0, sigma_max=0
    )
    assert isinstance(output, torch.Tensor)


# Test discretization
@import_or_fail("cftime")
@pytest.mark.parametrize("discretization", ["vp", "ve", "iddpm", "edm"])
def test_deterministic_sampler_discretization(mock_net, discretization, pytestconfig):

    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, discretization=discretization
    )
    assert isinstance(output, torch.Tensor)


# Test schedule
@import_or_fail("cftime")
@pytest.mark.parametrize("schedule", ["vp", "ve", "linear"])
def test_deterministic_sampler_schedule(mock_net, schedule, pytestconfig):

    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, schedule=schedule
    )
    assert isinstance(output, torch.Tensor)


# Test number of steps
@import_or_fail("cftime")
@pytest.mark.parametrize("num_steps", [1, 5, 18])
def test_deterministic_sampler_num_steps(mock_net, num_steps, pytestconfig):

    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net, latents=latents, img_lr=img_lr, num_steps=num_steps
    )
    assert isinstance(output, torch.Tensor)


# Test sigma
@import_or_fail("cftime")
@pytest.mark.parametrize("sigma_min, sigma_max", [(0.001, 0.01), (1.0, 1.5)])
def test_deterministic_sampler_sigma_boundaries(
    mock_net, sigma_min, sigma_max, pytestconfig
):

    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    output = deterministic_sampler(
        net=mock_net,
        latents=latents,
        img_lr=img_lr,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    assert isinstance(output, torch.Tensor)


# Test error handling
@import_or_fail("cftime")
@pytest.mark.parametrize("scaling", ["invalid_scaling", "vp", "none"])
def test_deterministic_sampler_scaling_validation(mock_net, scaling, pytestconfig):

    from physicsnemo.utils.diffusion import deterministic_sampler

    latents = torch.randn(1, 3, 64, 64)
    img_lr = torch.randn(1, 3, 64, 64)
    if scaling == "invalid_scaling":
        with pytest.raises(ValueError):
            deterministic_sampler(
                net=mock_net, latents=latents, img_lr=img_lr, scaling=scaling
            )
    else:
        output = deterministic_sampler(
            net=mock_net, latents=latents, img_lr=img_lr, scaling=scaling
        )
        assert isinstance(output, torch.Tensor)


# Test correctness with known ODE solution
@import_or_fail("cftime")
def test_deterministic_sampler_correctness(pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    # Create a simple network that implements our ODE: dx/dt = -x ==> x(t) = exp(-t)
    class SimpleODENet(torch.nn.Module):
        def __init__(self, sigma_min=0.002, sigma_max=4.0):
            super().__init__()
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max

        def forward(self, x, img_lr, sigma, class_labels=None):
            # Simulating ODE dx/dt = -x, we need denoiser to return (t + 1) * x
            # See EDM paper eqs. 3 and 4, and note EDM uses sigma <==> t
            return (sigma + 1.0) * x

        def round_sigma(self, sigma):
            return torch.tensor(sigma)

    # Create network and initial condition
    net = SimpleODENet()
    x0 = (
        torch.exp(-net.sigma_max * torch.ones(1, 1, 1, 1)) / net.sigma_max
    )  # "Initial condition" x(sigma_max) = exp(-sigma_max)
    img_lr = torch.zeros(1, 1, 1, 1)  # Dummy conditioning input

    # Run the sampler
    x_final = deterministic_sampler(
        net=net,
        latents=x0,
        img_lr=img_lr,
        num_steps=100,
        sigma_min=net.sigma_min,
        sigma_max=net.sigma_max,
    )

    # Analytical solution of x(t) = exp(-t) at t=0 is 1
    analytical_solution = torch.ones_like(x_final)

    # Check with loose tolerance since we're using numerical integration
    assert torch.allclose(
        x_final, analytical_solution, rtol=1e-2, atol=1e-2
    ), f"Numerical solution {x_final.item():.6f} does not match analytical solution {analytical_solution.item():.6f}"


# Mock network class with embedding_selector
class MockNet_embedding_selector:
    def __init__(self, sigma_min=0.1, sigma_max=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def round_sigma(self, t):
        return t

    def __call__(
        self,
        x,
        x_lr,
        t,
        class_labels,
        global_index=None,
        embedding_selector=None,
    ) -> torch.Tensor:
        # Mock behavior: return input tensor for testing purposes
        return x * 0.9


# The test function for patch-based deterministic_sampler
@import_or_fail("cftime")
def test_deterministic_sampler_args(pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    net = MockNet_embedding_selector()
    latents = torch.randn(2, 3, 448, 448)  # Mock latents
    img_lr = torch.randn(2, 3, 112, 112)  # Mock low-res image

    # Basic sampler functionality test
    result = deterministic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=None,
        mean_hr=None,
    )

    assert result.shape == latents.shape, "Output shape does not match expected shape"

    # Test with mean_hr conditioning
    mean_hr = torch.randn(2, 3, 112, 112)
    result_mean_hr = deterministic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=None,
        mean_hr=mean_hr,
        num_steps=2,
    )

    assert (
        result_mean_hr.shape == latents.shape
    ), "Mean HR conditioned output shape does not match expected shape"


# The test function for edm_sampler with rectangular domain and patching
@import_or_fail("cftime")
def test_deterministic_sampler_rectangle_patching(pytestconfig):
    from physicsnemo.utils.generative import deterministic_sampler
    from physicsnemo.utils.patching import GridPatching2D

    net = MockNet_embedding_selector()

    img_shape_y, img_shape_x = 256, 64
    patch_shape_y, patch_shape_x = 16, 10

    latents = torch.randn(2, 3, img_shape_y, img_shape_x)  # Mock latents
    img_lr = torch.randn(2, 3, img_shape_y, img_shape_x)  # Mock low-res image

    # Test with patching
    patching = GridPatching2D(
        img_shape=(img_shape_y, img_shape_x),
        patch_shape=(patch_shape_y, patch_shape_x),
        overlap_pix=4,
        boundary_pix=2,
    )

    # Test with mean_hr conditioning
    mean_hr = torch.randn(2, 3, img_shape_y, img_shape_x)
    result_mean_hr = deterministic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=patching,
        mean_hr=mean_hr,
        num_steps=2,
    )

    assert (
        result_mean_hr.shape == latents.shape
    ), "Mean HR conditioned output shape does not match expected shape"


# Mock network class with lead_time_embedding
class MockNet_lead_time_embedding:
    def __init__(self, sigma_min=0.1, sigma_max=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def round_sigma(self, t):
        return t

    def __call__(
        self,
        x,
        x_lr,
        t,
        class_labels,
        lead_time_label=None,
        global_index=None,
        embedding_selector=None,
    ) -> torch.Tensor:
        # Mock behavior: return input tensor for testing purposes
        return x


# The test function for patch-based deterministic_sampler with lead_time_embedding
@import_or_fail("cftime")
def test_deterministic_sampler_lead_time(pytestconfig):

    from physicsnemo.utils.generative import deterministic_sampler

    net = MockNet_lead_time_embedding()
    latents = torch.randn(2, 3, 448, 448)  # Mock latents
    img_lr = torch.randn(2, 3, 112, 112)  # Mock low-res image

    # Basic sampler functionality test
    result = deterministic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        patching=None,
        mean_hr=torch.ones_like(img_lr),
        lead_time_label=[0],
    )

    assert result.shape == latents.shape, "Output shape does not match expected shape"
