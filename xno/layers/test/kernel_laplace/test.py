import torch
import numpy as np
from torch import nn

from spectral_convolution_laplace import SpectralConvLaplace
from PR import PR
from PR2D import PR2d
from PR3D import PR3d

import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

# Set manual seeds for determinism in some tests
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define original Laplace classes (assumed imported)
# from your_code import PR, PR2d, PR3d
# from your_code import SpectralConvLaplace

def generate_grid(shape, device):
    """Generate coordinate grid for given shape."""
    return [torch.linspace(0, 1, s, device=device) for s in shape]

def compute_errors(ref, test):
    """Compute error metrics between ref (original) and test (spectral) outputs.
    
    Returns:
        dict: contains max error, mean abs error, relative L2 error
    """
    diff = ref - test
    max_error = torch.max(torch.abs(diff)).item()
    mean_abs_error = torch.mean(torch.abs(diff)).item()
    # relative L2 error: ||x-y||_2 / ||x||_2
    ref_norm = torch.norm(ref)
    rel_l2_error = (torch.norm(diff) / (ref_norm + 1e-12)).item()
    return {
        "max_error": max_error,
        "mean_abs_error": mean_abs_error,
        "relative_L2_error": rel_l2_error
    }

def run_test(dims, in_channels, out_channels, modes, shape, batch_size,
             complex_data=False, fft_norm='backward', seed=None):
    """Run a single test comparing original PR code vs SpectralConvLaplace.
    
    Args:
        dims (int): number of dimensions
        in_channels (int)
        out_channels (int)
        modes (list[int]): modes per dimension
        shape (tuple[int]): spatial dimension sizes
        batch_size (int)
        complex_data (bool)
        fft_norm (str)
        seed (int or None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Prepare input
    x = torch.randn(batch_size, in_channels, *shape, dtype=torch.float32, device=device)
    grid = generate_grid(shape, device=device)

    # Original code class
    # if dims == 1:
    #     original_conv = PR(in_channels, out_channels, modes[0]).to(device)
    # elif dims == 2:
    #     original_conv = PR2d(in_channels, out_channels, modes[0], modes[1]).to(device)
    # elif dims == 3:
    #     original_conv = PR3d(in_channels, out_channels, modes[0], modes[1], modes[2]).to(device)
    # else:
    #     raise ValueError("dims must be 1,2, or 3")

    # Run original
    # with torch.no_grad():
    #     original_output = original_conv(x)

    # Our spectral class
    spectral_conv = SpectralConvLaplace(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=modes,
        max_n_modes=modes,
        complex_data=complex_data,
        bias=True,
        separable=False,
        factorization=None,  # no factorization for simplicity
        implementation="reconstructed",
        fft_norm=fft_norm,
        device=device
    ).to(device)

    # Run spectral
    with torch.no_grad():
        spectral_output = spectral_conv(x, grid=grid)

    # Compute errors
    # errors = compute_errors(original_output, spectral_output)
    # return errors, original_output, spectral_output, x
    return spectral_output, x

# Prepare multiple test scenarios
# We vary dimensions, shapes, modes, and potentially complex_data
test_scenarios = [
    # Simple 1D tests
    {"dims": 1, "modes": [8], "shape": (16,), "batch": 2, "in_ch": 2, "out_ch": 2, "complex_data": False},
    {"dims": 1, "modes": [8], "shape": (32,), "batch": 4, "in_ch": 3, "out_ch": 3, "complex_data": False},
    {"dims": 1, "modes": [4], "shape": (16,), "batch": 1, "in_ch": 1, "out_ch": 1, "complex_data": False},
    # 2D tests
    {"dims": 2, "modes": [8, 8], "shape": (16, 16), "batch": 2, "in_ch": 2, "out_ch": 2, "complex_data": False},
    {"dims": 2, "modes": [4, 4], "shape": (8, 8),   "batch": 2, "in_ch": 3, "out_ch": 3, "complex_data": False},
    # 3D tests
    {"dims": 3, "modes": [4,4,4], "shape": (8,8,8), "batch": 1, "in_ch": 1, "out_ch": 1, "complex_data": False},
    {"dims": 3, "modes": [8,8,4], "shape": (16,16,8), "batch": 2, "in_ch": 2, "out_ch": 2, "complex_data": False},
]

# Run all tests, store results
all_results = []
for i, scenario in enumerate(test_scenarios):
    seed = 1234 + i
    errors, original_out, spectral_out, x_in = run_test(
        dims=scenario["dims"],
        in_channels=scenario["in_ch"],
        out_channels=scenario["out_ch"],
        modes=scenario["modes"],
        shape=scenario["shape"],
        batch_size=scenario["batch"],
        complex_data=scenario["complex_data"],
        fft_norm='backward',
        seed=seed
    )
    res = {
        "Test_ID": i,
        "Dimensions": scenario["dims"],
        "Shape": scenario["shape"],
        "Modes": scenario["modes"],
        "Batch": scenario["batch"],
        "In_Channels": scenario["in_ch"],
        "Out_Channels": scenario["out_ch"],
        "Max_Error": errors["max_error"],
        "Mean_Abs_Error": errors["mean_abs_error"],
        "Rel_L2_Error": errors["relative_L2_error"]
    }
    all_results.append(res)

# Create a DataFrame for better visualization of results
df = pd.DataFrame(all_results)
print("Summary of Test Results:")
print(df)

# Show summary statistics
print("\nStatistics:")
print(df[["Max_Error", "Mean_Abs_Error", "Rel_L2_Error"]].describe())

# Visualize a subset of results (e.g., first scenario) if dims=1 for debugging
# Only do this if 1D data and small shape.
for scenario_index, scenario in enumerate(test_scenarios):
    if scenario["dims"] == 1 and scenario["shape"][0] <= 32:
        # Run a single test again (no random this time)
        errors, original_out, spectral_out, x_in = run_test(
            dims=scenario["dims"],
            in_channels=scenario["in_ch"],
            out_channels=scenario["out_ch"],
            modes=scenario["modes"],
            shape=scenario["shape"],
            batch_size=scenario["batch"],
            complex_data=scenario["complex_data"],
            fft_norm='backward',
            seed=9999
        )
        # Visualize first batch, first channel output
        plt.figure()
        batch_idx = 0
        ch_idx = 0
        original_slice = original_out[batch_idx, ch_idx].cpu().detach().numpy()
        spectral_slice = spectral_out[batch_idx, ch_idx].cpu().detach().numpy()
        plt.plot(original_slice, label='Original')
        plt.plot(spectral_slice, label='Spectral', linestyle='--')
        plt.title(f"Visualization for Test {scenario_index}: {scenario['dims']}D")
        plt.xlabel('Spatial Index')
        plt.ylabel('Output Value')
        plt.legend()
        plt.show()
        # Just visualize one scenario
        break

