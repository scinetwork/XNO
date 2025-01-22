# test_xyno_block.py

import pytest
import torch
import torch.nn.functional as F
from itertools import combinations

##############################################
# Adjust the import to match your actual path:
# from xno.layers.xyno_block import XYNOBlocks
##############################################
from xno.layers.xyno_block import XYNOBlocks


###############################################################################
# Helpers
###############################################################################

def requires_wno_kwargs(kernels):
    """Check if any kernel in the given list is 'wno' 
    which requires wavelet parameters in transformation_kwargs."""
    return any(k.lower() == 'wno' for k in kernels)

def requires_lno_norm(kernels):
    """Check if any kernel is 'lno', which might auto-set or require group_norm, etc.
    This logic depends on your factory or design. 
    For the sake of example, we won't forcibly enforce norm changes. 
    But you might adapt it if your code auto-forces norm = 'group_norm' for LNO.
    """
    return any(k.lower() == 'lno' for k in kernels)

kernels_list = []
kernels = ['fno', 'wno', 'hno', 'lno']
for i in range(1, len(kernels) + 1):
    kernels_list.extend(combinations(kernels, i))

parametrize_kernels = [list(k) for k in kernels_list]

###############################################################################
# 1. Basic forward tests
###############################################################################
@pytest.mark.parametrize("in_channels,out_channels", [(3, 3), (4, 4), (2, 5)])
@pytest.mark.parametrize("n_modes", [(4,), (4, 4)])
# @pytest.mark.parametrize("kernels", [
#     ['fno'],                        # Single kernel
#     ['fno', 'wno'],                # Multiple kernels => parallel in one layer
#     ['hno'],                       # Single kernel
#     ['lno'],                       # Single kernel
# ])
@pytest.mark.parametrize("kernels", parametrize_kernels)

@pytest.mark.parametrize("mix_mode", ['parallel', 'pure'])
def test_xyno_block_basic_forward(in_channels, out_channels, n_modes, kernels, mix_mode):
    """
    Tests a forward pass on XYNOBlocks for different channels, dimensions, 
    kernel sets (old vs. new), and mix_mode. 
    """
    # transformation_kwargs
    # if we have WNO in the kernels, we need wavelet parameters in transformation_kwargs:
    transformation_kwargs = None
    if requires_wno_kwargs(kernels):
        # minimal stub wavelet config
        # e.g. wavelet_size matches the dimension from n_modes
        size = [16] * len(n_modes)
        transformation_kwargs = {
            "wavelet_level": 2,
            "wavelet_size": size,
            "wavelet_filter": ["db4"],
            "wavelet_mode": "symmetric"
        }

    dim = len(n_modes)
    batch_size = 2
    spatial_size = [16] * dim
    x = torch.randn(batch_size, in_channels, *spatial_size)

    block = XYNOBlocks(
        in_channels      = in_channels,
        out_channels     = out_channels,
        n_modes          = n_modes,
        kernels          = kernels,         # e.g. ['fno', 'wno']
        mix_mode         = mix_mode,        # 'parallel' or 'pure'
        transformation_kwargs = transformation_kwargs,
        n_layers         = 1,               # single layer for simplicity
        channel_mlp_skip = 'linear' if in_channels != out_channels else 'soft-gating',
    )
    
    if mix_mode == 'parallel':
        y = block(x)
    else:
        for kernel in kernels:
            y = block(x, kernel=kernel)
        
    # Forward pass
    # y = block(x)
    # Output shape check
    assert y.shape[:2] == (batch_size, out_channels), (
        f"Channels mismatch. Got {y.shape[:2]}, expected {(batch_size, out_channels)}."
    )
    assert y.shape[2:] == tuple(spatial_size), (
        f"Spatial shape mismatch. Got {y.shape[2:]}, expected {spatial_size}."
    )

###############################################################################
# 2. Unknown kernel
###############################################################################
def test_xyno_block_unknown_kernel():
    """
    Passing an unknown kernel name should raise a ValueError.
    """
    with pytest.raises(ValueError):
        block = XYNOBlocks(
            in_channels  = 3,
            out_channels = 3,
            n_modes      = [4, 4],
            kernels      = ['fno', 'unknown_kernel'],  # 'unknown_kernel' is not valid
        )
        x = torch.randn(2, 3, 16, 16)
        block(x)  # Should raise error internally.


###############################################################################
# 3. WNO missing transformation_kwargs
###############################################################################
def test_xyno_block_missing_transformation_kwargs_for_wno():
    """
    If 'wno' kernel is present but wavelet parameters are not given, 
    we expect a ValueError.
    """
    with pytest.raises(ValueError):
        block = XYNOBlocks(
            in_channels  = 3,
            out_channels = 3,
            n_modes      = [4,4],
            kernels      = ['wno'],  # WNO requires wavelet param
            # transformation_kwargs=None => Should raise
        )
        x = torch.randn(2, 3, 16, 16)
        block(x)


###############################################################################
# 4. Resolution scaling factor
###############################################################################
@pytest.mark.parametrize("scaling_factor", [0.5, 2, None])  # Mixed scalars / lists
@pytest.mark.parametrize("dim", [1, 2])
def test_xyno_block_resolution_scaling_factor(scaling_factor, dim):
    """
    Test upsampling / downsampling logic for different dimensional data.
    We'll only use a single kernel for simplicity. 
    """
    n_modes = [4]*dim
    size = [10]*dim
    x = torch.randn(2, 3, *size)
    
    block = XYNOBlocks(
        in_channels  = 3,
        out_channels = 4,
        n_modes      = n_modes,
        kernels      = ['fno'],     # Single kernel
        mix_mode     = 'parallel',  # not relevant for single kernel
        resolution_scaling_factor = scaling_factor,
        n_layers     = 1,
        channel_mlp_skip='linear'
    )
    y = block(x)
    
    # If scaling_factor is None -> shape remains the same
    # If scaling_factor is a float or int, we scale each dimension
    if scaling_factor is None:
        assert list(y.shape) == [2, 4, *size], (
            f"Expected shape [2,4,{size}], got {list(y.shape)}."
        )
    elif isinstance(scaling_factor, (float, int)):
        expected_size = [int(s * scaling_factor) for s in size]
        assert list(y.shape) == [2, 4, *expected_size], (
            f"Expected [2,4,{expected_size}], got {list(y.shape)}."
        )
    else:
        # e.g. [0.5, 2], but if dim=1 we can only apply the first entry 
        # Here, your code might either handle or throw an error 
        # depending on how validate_scaling_factor is implemented 
        if dim == 1:
            factor = scaling_factor[0]
            expected_size = [int(size[0] * factor)]
        else:
            # If dim=2, then scale each dimension by the respective factor
            expected_size = [
                int(size[i] * scaling_factor[i]) for i in range(dim)
            ]
        assert list(y.shape) == [2, 4, *expected_size], (
            f"Expected [2,4,{expected_size}], got {list(y.shape)}."
        )

###############################################################################
# 5. SubModule / get_block tests
###############################################################################
def test_xyno_block_submodule():
    """
    If multiple layers, test retrieving a sub-block for a specific layer 
    via get_block. 
    """
    block = XYNOBlocks(
        in_channels  = 4,
        out_channels = 4,
        n_modes      = [4, 4],
        kernels      = ['fno', 'lno'],
        mix_mode     = 'pure',
        n_layers     = 3  # multiple layers
    )
    x = torch.randn(2, 4, 16, 16)
    
    # get the second block (index=1)
    sub_block = block.get_block(indices=1)
    y = sub_block(x)
    
    # shape must be (2, out_channels, 16, 16)
    assert y.shape == (2, 4, 16, 16)


def test_xyno_block_submodule_single_layer_raises():
    """
    If only one layer is present, get_block should raise a ValueError 
    per the design.
    """
    block = XYNOBlocks(
        in_channels  = 3,
        out_channels = 4,
        n_modes      = [4, 4],
        kernels      = ['fno'],
        mix_mode     = 'pure',
        n_layers     = 1,
        channel_mlp_skip='linear'
    )
    with pytest.raises(ValueError):
        block.get_block(indices=0)


###############################################################################
# 6. Preactivation test
###############################################################################
@pytest.mark.parametrize("kernels", [
    ['fno'],
    ['fno', 'wno'],
    ['lno'],
])
def test_xyno_block_preactivation(kernels):
    """
    If preactivation=True, we call forward_with_preactivation.
    Just ensure no crash and shape correctness.
    """
    transformation_kwargs = None
    if requires_wno_kwargs(kernels):
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": [8, 8],
            "wavelet_filter": ["db4"],
            "wavelet_mode": "symmetric"
        }
    block = XYNOBlocks(
        in_channels  = 3,
        out_channels = 3,
        n_modes      = [4, 4],
        kernels      = kernels,
        mix_mode     = 'parallel',
        transformation_kwargs = transformation_kwargs,
        preactivation= True,
        n_layers     = 2
    )
    x = torch.randn(2, 3, 16, 16)
    y = block(x)
    assert y.shape == (2, 3, 16, 16)


###############################################################################
# 7. Tanh stabilizer
###############################################################################
def test_xyno_block_tanh_stabilizer():
    """
    If stabilizer='tanh', apply tanh to input before the conv in forward.
    """
    block = XYNOBlocks(
        in_channels  = 2,
        out_channels = 2,
        n_modes      = [4],
        kernels      = ['fno'],
        mix_mode     = 'parallel',
        n_layers     = 1,
        stabilizer   = 'tanh',
    )
    x = torch.randn(2, 2, 16)
    y = block(x)
    assert y.shape == (2, 2, 16)


###############################################################################
# 8. Multiple layers, "pure" mode, distinct kernel picks at each layer
###############################################################################
def test_xyno_block_pure_distinct_kernels_each_layer():
    """
    If we have multiple layers with mix_mode='pure', the forward pass uses 
    a single kernel from the 'kernels' list for each layer in order.
    For example, if kernels = ['fno','wno','lno'], then:
       - layer 0 uses 'fno'
       - layer 1 uses 'wno'
       - layer 2 uses 'lno'
    We'll confirm that the forward pass runs with correct shape 
    and wavelet params for 'wno' if needed.
    """
    # kernels used for 3 layers
    kernels = ['fno', 'wno', 'lno']
    # We must supply wavelet params for 'wno'
    transformation_kwargs = {
        "wavelet_level": 1,
        "wavelet_size": [8],  # for 1D case
        "wavelet_filter": ["db4"],
        "wavelet_mode": "symmetric"
    }
    block = XYNOBlocks(
        in_channels  = 2,
        out_channels = 2,
        n_modes      = [4],  # 1D
        kernels      = kernels,
        mix_mode     = 'pure',
        transformation_kwargs = transformation_kwargs,
        n_layers     = 3,
    )
    x = torch.randn(2, 2, 16)
    y = block(x)  # Should run fno at layer0, wno at layer1, lno at layer2
    assert y.shape == (2, 2, 16)
