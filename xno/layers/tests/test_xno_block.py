# test_xno_block.py

import pytest
import torch
import torch.nn.functional as F

# Adjust this import to match your actual package/module layout
# from xno.layers.xno_block import XNOBlocks
from ..xno_block import XNOBlocks


@pytest.mark.parametrize("transformation", ["FNO", "HNO", "LNO", "WNO"])
@pytest.mark.parametrize("in_channels,out_channels", [(3, 4), (4, 4)])
@pytest.mark.parametrize("n_modes", [(4,), (4, 4), (4, 4, 4)])
def test_XNOBlock_basic_forward(transformation, in_channels, out_channels, n_modes):
    """
    Basic forward test for XNOBlocks with different transformations, channels, 
    and dimensions. Ensures:
      - No runtime errors
      - Output shape matches input shape
      - The transform can handle different n_dim
    """
    # For test simplicity, we keep a single layer
    n_layers = 1

    # Deduce the dimension from n_modes
    dim = len(n_modes)
    # Make input shape
    size = [10] * dim  # e.g. 10 for each dimension
    x = torch.randn(2, in_channels, *size)

    # If using WNO or LNO, we might need minimal dummy transformation_kwargs
    # For example, wavelet or laplace might require "wavelet_size" or "wavelet_level".
    transformation_kwargs = {}
    if transformation.lower() == "wno":
        # Minimal stub. Adjust to your wavelet dimension requirements.
        # If dim=2, wavelet_size=[10,10], wavelet_level=2, e.g.
        wavelet_size = size[:]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size,
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
    elif transformation.lower() == "lno":
        # Possibly no extra arguments needed, or you might configure something 
        # for Laplace1D/2D/3D
        transformation_kwargs = {}
        # in_channels = out_channels
    elif transformation.lower() == "hno":
        # Hilbert might require none or some optional args
        transformation_kwargs = {}
    
    if in_channels != out_channels: channel_mlp_skip='linear'
    else: channel_mlp_skip = "soft-gating"

    block = XNOBlocks(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs,
        n_layers=n_layers,
        channel_mlp_skip=channel_mlp_skip
    )

    y = block(x)
    # The output shape should match (batch=2, out_channels, *size)
    assert y.shape == (2, out_channels, *size), (
        f"Output shape mismatch. Expected {(2, out_channels, *size)}, got {y.shape}."
    )


def test_XNOBlock_unknown_transformation():
    """
    If user passes an unknown string for 'transformation', we expect an error.
    """
    with pytest.raises(ValueError):
        _ = XNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=[4, 4],
            transformation="UNKNOWN_TRANSFORM",
        )


def test_XNOBlock_missing_transformation_kwargs_for_WNO():
    """
    If transformation='WNO' but 'transformation_kwargs' is missing required
    wavelet parameters, we expect a ValueError (depending on the factory logic).
    """
    with pytest.raises(ValueError):
        _ = XNOBlocks(
            in_channels=3,
            out_channels=3,
            n_modes=[4, 4],
            transformation="WNO",
            # transformation_kwargs=None => Should raise
        )


def test_XNOBlock_resolution_scaling_factor():
    """
    Adaptation of the FNOBlock scaling-factor test to XNOBlocks.
    Checks up-/downsampling for different dimensions.
    """
    max_n_modes = [8, 8, 8, 8]
    n_modes = [4, 4, 4, 4]
    size = [10] * 4  # up to 4D

    for dim in [1, 2, 3, 4]:
        block = XNOBlocks(
            3, 4, max_n_modes[:dim], 
            max_n_modes=max_n_modes[:dim], n_layers=1, 
            transformation="FNO", 
            channel_mlp_skip="linear"
        )

        # Check setter logic for n_modes property
        block.n_modes = n_modes[:dim]
        # (Within the block, each spectral conv might internally 
        #  do real-> N//2 +1 in the last dimension if it's 'Fourier', 
        #  but we skip details here unless we want a direct check.)

        # Downsample outputs
        block = XNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=n_modes[:dim],
            n_layers=1,
            resolution_scaling_factor=0.5,
            transformation="FNO",
            channel_mlp_skip="linear"
        )

        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        expected_down = [m // 2 for m in size[:dim]]
        assert list(res.shape[2:]) == expected_down, (
            f"Expected downsample shape {expected_down}, got {res.shape[2:]}"
        )

        # Upsample outputs
        block = XNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=n_modes[:dim],
            n_layers=1,
            resolution_scaling_factor=2,
            transformation="FNO",
            channel_mlp_skip="linear"
        )
        x = torch.randn(2, 3, *size[:dim])
        res = block(x)
        expected_up = [m * 2 for m in size[:dim]]
        assert res.shape[1] == 4, "Check out_channels"
        assert list(res.shape[2:]) == expected_up, (
            f"Expected upsample shape {expected_up}, got {res.shape[2:]}"
        )


@pytest.mark.parametrize('norm', ['instance_norm', 'ada_in', 'group_norm', None])
def test_XNOBlock_norm(norm):
    """
    Tests XNOBlocks with various norm layers. 
    If norm='ada_in', we set embeddings and ensure no errors.
    """
    modes = (8, 8)
    size = [10, 10]
    ada_in_features = 4

    block = XNOBlocks(
        in_channels=3,
        out_channels=4,
        n_modes=modes,
        n_layers=1,
        norm=norm,
        ada_in_features=ada_in_features,
        transformation="FNO",  # or "LNO", etc.
        channel_mlp_skip="linear"
    )

    if norm == 'ada_in':
        embedding = torch.randn(ada_in_features)
        block.set_ada_in_embeddings(embedding)

    x = torch.randn(2, 3, *size)
    res = block(x)
    assert list(res.shape[2:]) == size, (
        f"Expected shape {size} in spatial dims, got {res.shape[2:]}"
    )


@pytest.mark.parametrize('n_dim', [1, 2, 3])
def test_XNOBlock_complex_data(n_dim):
    """
    Test XNO with complex input data, verifying shape correctness
    and that no errors arise if 'complex_data=True'.
    """
    modes = (8, 8, 8)
    size = [10] * 3
    x = torch.randn(2, 3, *size[:n_dim], dtype=torch.cfloat)
    
    transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": [2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }

    block = XNOBlocks(
        in_channels=3,
        out_channels=4,
        n_modes=modes[:n_dim],
        n_layers=2,
        complex_data=True,
        transformation="FNO",  # Just for FNO
        channel_mlp_skip="linear", 
    )
    res = block(x)
    assert list(res.shape[2:]) == size[:n_dim], (
        f"Expected shape {size[:n_dim]}, got {res.shape[2:]}"
    )


def test_XNOBlock_submodule():
    """
    Test get_block() submodule logic. If we have multiple layers,
    we can retrieve a specific sub-block and run it.
    """
    block = XNOBlocks(
        in_channels=3,
        out_channels=4,
        n_modes=[4, 4],
        n_layers=3,  # multiple layers
        transformation="HNO",
        channel_mlp_skip="linear"
    )
    x = torch.randn(2, 3, 10, 10)
    # get the second block
    sub = block.get_block(indices=1)
    y = sub(x)
    # The sub-block forward returns shape same as the main block but only 
    # processes the 'index=1' layer. 
    assert y.shape == (2, 4, 10, 10)

@pytest.mark.parametrize("transformation", ["FNO", "HNO", "LNO", "WNO"])
def test_XNOBlock_preactivation(transformation):
    """
    If 'preactivation=True', the forward pass calls 'forward_with_preactivation'.
    We verify no crash and shape correctness.
    """
    transformation_kwargs = {}
    
    if transformation == "WNO":
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": [2, 2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
    block = XNOBlocks(
        in_channels=3,
        out_channels=3,
        n_modes=[4, 4],
        n_layers=2,
        preactivation=True,
        transformation=transformation, 
        channel_mlp_skip="linear", 
        transformation_kwargs=transformation_kwargs
    )
    x = torch.randn(2, 3, 10, 10)
    res = block(x)
    assert res.shape == (2, 3, 10, 10)
    # No numerical checks but at least we confirm it runs.


def test_XNOBlock_tanh_stabilizer():
    """
    If stabilizer='tanh', we apply tanh to the input 
    before the spectral conv in forward.
    """
    block = XNOBlocks(
        in_channels=2,
        out_channels=2,
        n_modes=[4],
        n_layers=1,
        stabilizer="tanh",
        transformation="FNO"
    )
    x = torch.randn(2, 2, 16)
    # Just ensure forward completes without error
    y = block(x)
    assert y.shape == (2, 2, 16)
