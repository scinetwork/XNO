
import pytest
import torch
import numpy as np

from xno.layers.spectral_convolution_wavelet import (
    SpectralConvWavelet1D, 
    SpectralConvWavelet2D, 
    SpectralConvWavelet2DCwt, 
    SpectralConvWavelet3D
)


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("wavelet_level", [1, 2, 4])
@pytest.mark.parametrize("wavelet_size", [[8], [16], [24]])
@pytest.mark.parametrize("wavelet_filter", [["db1"], ["db2"], ["db4"]])
@pytest.mark.parametrize("wavelet_mode", ["symmetric", "zero", "periodization"])
def test_wavelet1d_forward(
    in_channels, 
    out_channels, 
    wavelet_level, 
    wavelet_size, 
    wavelet_filter, 
    wavelet_mode
):
    """
    Test the basic forward pass for SpectralConvWavelet1D using different valid parameter combos.

    Checks:
      - No runtime errors
      - Output shape matches the input shape (since we do IDWT back)
      - Real output for real input
    """
    batch_size = 2
    length = wavelet_size[0]  # We match the test input length to wavelet_size
    conv = SpectralConvWavelet1D(
        in_channels=in_channels,
        out_channels=out_channels,
        wavelet_level=wavelet_level,
        wavelet_size=wavelet_size,
        wavelet_filter=wavelet_filter,  # e.g. ['db4']
        wavelet_mode=wavelet_mode,
    )

    x = torch.randn(batch_size, in_channels, length)
    y = conv(x)
    assert y.shape == (batch_size, out_channels, length), (
        f"Expected output shape ({batch_size},{out_channels},{length}), "
        f"got {y.shape}."
    )
    assert torch.is_floating_point(y), "Output should be real-valued float Tensor."


@pytest.mark.parametrize("wavelet_size", [[8], [16]])
@pytest.mark.parametrize("resolution_scaling_factor", [0.5, 2.0])
def test_wavelet1d_transform(wavelet_size, resolution_scaling_factor):
    """
    Test the transform() method, verifying up/downsampling shape changes.
    """
    conv = SpectralConvWavelet1D(
        in_channels=1,
        out_channels=1,
        wavelet_level=2,
        wavelet_size=wavelet_size,
        wavelet_filter=["db2"],
        wavelet_mode="symmetric",
        resolution_scaling_factor=resolution_scaling_factor,
    )
    batch_size = 1
    length = wavelet_size[0]
    x = torch.randn(batch_size, 1, length)

    x_t = conv.transform(x)
    expected_length = round(length * resolution_scaling_factor)
    assert x_t.shape == (batch_size, 1, expected_length), (
        f"Expected {expected_length} in last dimension, got {x_t.shape[-1]}."
    )


@pytest.mark.parametrize("in_size,wave_size", [
    # Input bigger than wavelet_size
    (32, 16),
    # Input smaller than wavelet_size
    (8, 16),
])
def test_wavelet1d_bigger_smaller_input(in_size, wave_size):
    """
    If input's last dimension is bigger/smaller than wavelet_size:
      - We adjust wavelet_level internally (in forward) by factor= log2(...)
      - Ensure the forward pass doesn't crash and final shape is consistent.
    """
    conv = SpectralConvWavelet1D(
        in_channels=2,
        out_channels=2,
        wavelet_level=2,
        wavelet_size=[wave_size],    # e.g. wave_size=16
        wavelet_filter=["db4"],
        wavelet_mode="symmetric",
    )

    x = torch.randn(2, 2, in_size)  # e.g. in_size=32
    y = conv(x)
    # The forward pass does an IDWT, so output shape = input shape => (2, 2, in_size)
    assert y.shape == (2, 2, in_size), f"Output shape mismatch: {y.shape}"


def test_wavelet1d_invalid_wavelet_size():
    """
    wavelet_size must be a list of exactly one element in this 1D implementation.
    Raise an Exception if user provides something else.
    """
    with pytest.raises(Exception):
        _ = SpectralConvWavelet1D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[8, 16],  # Wrong: 2 elements for 1D
            wavelet_filter=["db4"],
            wavelet_mode="symmetric",
        )

    with pytest.raises(Exception):
        _ = SpectralConvWavelet1D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=(8,),     # Wrong: a tuple, not a list
            wavelet_filter=["db4"],
            wavelet_mode="symmetric",
        )


@pytest.mark.parametrize("wave_filter", [["db4"], ["sym5"], ["haar"]])
def test_wavelet1d_forward_different_filters(wave_filter):
    """
    Just verify that different recognized wavelet filters can be used 
    without crashing. 
    """
    conv = SpectralConvWavelet1D(
        in_channels=1,
        out_channels=1,
        wavelet_level=2,
        wavelet_size=[16],
        wavelet_filter=wave_filter,   # e.g. ["db4"] or ["haar"]
        wavelet_mode="symmetric",
    )
    x = torch.randn(1, 1, 16)
    y = conv(x)
    assert y.shape == (1, 1, 16)


def test_wavelet1d_unrecognized_filter():
    """
    If user specifies an unknown wavelet_filter, 
    we might expect an error from the underlying wavelet library 
    (depending on how the DWT1D constructor behaves). 
    """
    with pytest.raises(Exception):
        _ = SpectralConvWavelet1D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[8],
            wavelet_filter=["nonexistent_wavelet"],
            wavelet_mode="symmetric",
        )


def test_wavelet1d_unrecognized_mode():
    """
    If user specifies an unrecognized wavelet_mode,
    we expect an error from the underlying library or the code itself.
    """
    with pytest.raises(Exception):
        _ = SpectralConvWavelet1D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[8],
            wavelet_filter=["db4"],
            wavelet_mode="some_unknown_mode",
        )


@pytest.mark.parametrize("wavelet_level", [1, 3, 6])
def test_wavelet1d_edge_wavelet_levels(wavelet_level):
    """
    Check extremes for wavelet_level, including 0 (which might mean no decomposition)
    and a level that is large relative to wavelet_size. 
    Ensure no crash occurs.
    """
    conv = SpectralConvWavelet1D(
        in_channels=1,
        out_channels=1,
        wavelet_level=wavelet_level,
        wavelet_size=[8],
        wavelet_filter=["db4"],
        wavelet_mode="symmetric",
    )
    x = torch.randn(1, 1, 8)
    y = conv(x)
    assert y.shape == (1, 1, 8), f"Unexpected shape for wavelet_level={wavelet_level}."


def test_wavelet1d_weight_shapes():
    """
    Ensure the internal weight shape is as expected:
      shape = (2, in_channels, out_channels, modes1)
    where modes1 is determined after the dummy wavelet transform on wavelet_size.
    """
    in_channels, out_channels = 2, 3
    wavelet_size = [16]
    wavelet_level = 2

    conv = SpectralConvWavelet1D(
        in_channels=in_channels,
        out_channels=out_channels,
        wavelet_level=wavelet_level,
        wavelet_size=wavelet_size,
        wavelet_filter=["db4"],
        wavelet_mode="symmetric",
    )
    W = conv.weight
    # 2 is for: 0-th index => low-pass weight, 1 => high-pass weight
    # in_channels => dimension 1
    # out_channels => dimension 2
    # last => self.modes1
    assert W.dim() == 4, f"Weight must be a 4D Tensor, got shape {W.shape}."
    assert W.shape[0] == 2, f"First dimension=2 for low/high pass, got {W.shape[0]}"
    assert W.shape[1] == in_channels, f"Expected in_channels={in_channels}, got {W.shape[1]}"
    assert W.shape[2] == out_channels, f"Expected out_channels={out_channels}, got {W.shape[2]}"
    # W.shape[3] should match self.modes1 (the approximate-coeff length from dummy_data)
    # We won't test the exact number, but confirm it's > 0.
    assert W.shape[3] > 0, "Expected non-zero wavelet mode dimension."


def test_wavelet1d_mul1d():
    """
    Direct test of the mul1d method:
      mul1d(input, weights) => Einstein sum "bix, iox -> box"
    Ensures correct shape logic for a simpler usage scenario.
    """
    conv = SpectralConvWavelet1D(
        in_channels=2,
        out_channels=3,
        wavelet_level=1,
        wavelet_size=[8],
        wavelet_filter=["db2"],
        wavelet_mode="symmetric",
    )
    # input: (batch=2, in_channels=2, x=5)
    input_tensor = torch.randn(2, 2, 5)
    # weights: (in_channels=2, out_channels=3, x=5)
    weight_tensor = torch.randn(2, 3, 5)
    
    result = conv.mul1d(input_tensor, weight_tensor)
    # shape => (batch=2, out_channels=3, x=5)
    assert result.shape == (2, 3, 5)


def test_wavelet1d_negative_test_wavelet_size_type():
    """
    If wavelet_size is not even a list, we expect an immediate Exception
    from the constructor.
    """
    with pytest.raises(Exception):
        _ = SpectralConvWavelet1D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size="16",  # Not a list => should raise
            wavelet_filter=["db4"],
            wavelet_mode="symmetric",
        )
