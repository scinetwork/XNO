
import pytest
import torch
import numpy as np

from xno.layers.spectral_convolution_wavelet import (
    SpectralConvWavelet1D, 
    SpectralConvWavelet2D, 
    SpectralConvWavelet2DCwt, 
    SpectralConvWavelet3D
)

# -------------------------------------------------------------------------
# TEST SUITE 1: SpectralConvWavelet1D
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# TEST SUITE 2: SpectralConvWavelet2D
# -------------------------------------------------------------------------
@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("wavelet_level", [1, 2, 3])
@pytest.mark.parametrize("wavelet_size", [[8, 8], [16, 12]])
@pytest.mark.parametrize("wavelet_filter", [["db2"], ["db4"], ["haar"]])
@pytest.mark.parametrize("wavelet_mode", ["symmetric", "zero", "periodization"])
def test_wavelet2d_forward(
    in_channels,
    out_channels,
    wavelet_level,
    wavelet_size,
    wavelet_filter,
    wavelet_mode
):
    """
    Basic forward pass test for SpectralConvWavelet2D with various parameters.

    Checks:
      - No runtime errors
      - Output shape matches input shape
      - Output is real float if input is real
    """
    batch_size = 2
    height, width = wavelet_size
    conv = SpectralConvWavelet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        wavelet_level=wavelet_level,
        wavelet_size=wavelet_size,
        wavelet_filter=wavelet_filter,
        wavelet_mode=wavelet_mode,
    )

    x = torch.randn(batch_size, in_channels, height, width)
    y = conv(x)
    assert y.shape == (batch_size, out_channels, height, width), (
        f"Expected output shape {batch_size, out_channels, height, width}, "
        f"got {y.shape}."
    )
    assert torch.is_floating_point(y), "Output must be a real-valued float Tensor."


@pytest.mark.parametrize("resolution_scaling_factor", [[0.5, 2.0], 1.5, 2.0])
def test_wavelet2d_transform(resolution_scaling_factor):
    """
    Test the transform() method for up/down-sampling in 2D.
    Checks that the shape changes as expected.
    """
    conv = SpectralConvWavelet2D(
        in_channels=1,
        out_channels=1,
        wavelet_level=2,
        wavelet_size=[16, 16],
        wavelet_filter=["db2"],
        wavelet_mode="symmetric",
        resolution_scaling_factor=resolution_scaling_factor,
    )
    x = torch.randn(1, 1, 16, 16)
    x_t = conv.transform(x)
    
    # Handle three possible types: a single float, a list of two floats, etc.
    if isinstance(resolution_scaling_factor, (float, int)):
        # uniform scaling for both dimensions
        expected_height = round(16 * resolution_scaling_factor)
        expected_width = round(16 * resolution_scaling_factor)
    elif isinstance(resolution_scaling_factor, (list, tuple)) and len(resolution_scaling_factor) == 2:
        expected_height = round(16 * resolution_scaling_factor[0])
        expected_width = round(16 * resolution_scaling_factor[1])
    else:
        raise ValueError("Unsupported resolution_scaling_factor format in test.")

    assert x_t.shape[-2:] == (expected_height, expected_width), (
        f"Expected shape (H={expected_height},W={expected_width}), got {x_t.shape[-2:]}"
    )


@pytest.mark.parametrize("in_shape,wave_shape", [
    # Input bigger than wavelet_size in width
    ((1, 2, 16, 32), [16, 16]),
    # Input smaller than wavelet_size in width
    ((1, 2, 16, 8), [16, 16]),
])
def test_wavelet2d_size_mismatch(in_shape, wave_shape):
    """
    If input last dimension is bigger/smaller than wavelet_size[-1],
    the code adjusts wavelet_level internally. 
    Ensure no crash and final output shape matches input shape.
    """
    conv = SpectralConvWavelet2D(
        in_channels=in_shape[1],
        out_channels=3,
        wavelet_level=2,
        wavelet_size=wave_shape,
        wavelet_filter=["db4"],
        wavelet_mode="symmetric",
    )

    x = torch.randn(*in_shape)
    y = conv(x)
    assert y.shape == (in_shape[0], 3, in_shape[2], in_shape[3])


def test_wavelet2d_invalid_wavelet_size():
    """
    wavelet_size must be a list of exactly two elements for 2D,
    or the constructor raises an Exception.
    """
    # Wrong length
    with pytest.raises(Exception, match="WaveConv2d accepts the wavelet_size of 2D signal"):
        _ = SpectralConvWavelet2D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[16],  # Only one element
            wavelet_filter=["db2"],
            wavelet_mode="symmetric",
        )

    # Not even a list
    with pytest.raises(Exception):
        _ = SpectralConvWavelet2D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=(16, 16),  # Tuple, not a list
            wavelet_filter=["db2"],
            wavelet_mode="symmetric",
        )


@pytest.mark.parametrize("wavelet_filter", [["db4"], ["sym5"], ["haar"]])
def test_wavelet2d_forward_different_filters(wavelet_filter):
    """
    Verify different recognized wavelet filters in 2D don't crash.
    """
    conv = SpectralConvWavelet2D(
        in_channels=2,
        out_channels=2,
        wavelet_level=2,
        wavelet_size=[16, 16],
        wavelet_filter=wavelet_filter,
        wavelet_mode="symmetric",
    )
    x = torch.randn(1, 2, 16, 16)
    y = conv(x)
    assert y.shape == (1, 2, 16, 16)


def test_wavelet2d_weight_shape():
    """
    Check that the internal weight shape is correct for 2D:
      shape = (4, in_channels, out_channels, modes1, modes2)
      4 => approximate, horiz-detail, vert-detail, diag-detail wavelets
    """
    conv = SpectralConvWavelet2D(
        in_channels=2,
        out_channels=3,
        wavelet_level=2,
        wavelet_size=[16, 16],
        wavelet_filter=["db4"],
        wavelet_mode="symmetric",
    )
    W = conv.weight
    assert W.dim() == 5, f"Expected 5D weight, got shape {W.shape}."
    assert W.shape[0] == 4, "First dim=4 wavelet subbands for 2D (LL, LH, HL, HH)."
    assert W.shape[1] == 2, "Mismatch in in_channels."
    assert W.shape[2] == 3, "Mismatch in out_channels."
    assert W.shape[3] > 0 and W.shape[4] > 0, "modes1 and modes2 must be > 0."


def test_wavelet2d_mul2d():
    """
    Direct test of the mul2d method:
      mul2d(input, weights) => "bixy, ioxy -> boxy"
    """
    conv = SpectralConvWavelet2D(
        in_channels=2,
        out_channels=2,
        wavelet_level=1,
        wavelet_size=[8, 8],
        wavelet_filter=["db2"],
        wavelet_mode="symmetric",
    )
    # input: (batch=2, in_channels=2, height=5, width=4)
    input_tensor = torch.randn(2, 2, 5, 4)
    # weights: (in_channels=2, out_channels=2, height=5, width=4)
    weight_tensor = torch.randn(2, 2, 5, 4)

    result = conv.mul2d(input_tensor, weight_tensor)
    assert result.shape == (2, 2, 5, 4), f"Got shape {result.shape}."


@pytest.mark.parametrize("wavelet_level", [1, 5])
def test_wavelet2d_edge_wavelet_levels(wavelet_level):
    """
    Check extremes for wavelet_level in 2D: 0 or large relative to wavelet_size.
    """
    conv = SpectralConvWavelet2D(
        in_channels=2,
        out_channels=2,
        wavelet_level=wavelet_level,
        wavelet_size=[8, 8],
        wavelet_filter=["db4"],
        wavelet_mode="symmetric",
    )
    x = torch.randn(1, 2, 8, 8)
    y = conv(x)
    assert y.shape == (1, 2, 8, 8)

# -------------------------------------------------------------------------
# TEST SUITE 3: SpectralConvWavelet2DCwt (dual-tree continuous wavelet)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("wavelet_level", [1, 2])
@pytest.mark.parametrize("wavelet_size", [[8, 8], [12, 10]])
@pytest.mark.parametrize(
    "wavelet_filter", [
        ["near_sym_b", "qshift_b"],  # typical
        ["antonini", "qshift_06"],   # alternative
    ]
)
def test_wavelet2dcwt_forward(
    in_channels,
    out_channels,
    wavelet_level,
    wavelet_size,
    wavelet_filter
):
    """
    Basic forward pass test for SpectralConvWavelet2DCwt (dual-tree).
      - Ensures no crash
      - Checks shape consistency
    """
    batch_size = 2
    h, w = wavelet_size
    conv = SpectralConvWavelet2DCwt(
        in_channels=in_channels,
        out_channels=out_channels,
        wavelet_level=wavelet_level,
        wavelet_size=wavelet_size,
        wavelet_filter=wavelet_filter,
    )
    x = torch.randn(batch_size, in_channels, h, w)
    y = conv(x)
    assert y.shape == (batch_size, out_channels, h, w), (
        f"Expected {batch_size,out_channels,h,w}, got {y.shape}"
    )
    assert torch.is_floating_point(y)


@pytest.mark.parametrize("resolution_scaling_factor", [0.5, 2.0, [0.75, 1.25]])
def test_wavelet2dcwt_transform(resolution_scaling_factor):
    """
    Test transform method in 2D dual-tree wavelet, checking up/down sampling.
    """
    conv = SpectralConvWavelet2DCwt(
        in_channels=1,
        out_channels=1,
        wavelet_level=1,
        wavelet_size=[10, 10],
        wavelet_filter=["near_sym_b", "qshift_b"],
        resolution_scaling_factor=resolution_scaling_factor,
    )
    x = torch.randn(1, 1, 10, 10)
    x_t = conv.transform(x)

    if isinstance(resolution_scaling_factor, (float, int)):
        expected_h = round(10 * resolution_scaling_factor)
        expected_w = round(10 * resolution_scaling_factor)
    else:  # list or tuple of length 2
        expected_h = round(10 * resolution_scaling_factor[0])
        expected_w = round(10 * resolution_scaling_factor[1])

    assert x_t.shape[-2:] == (expected_h, expected_w), "transform shape mismatch."


@pytest.mark.parametrize("in_shape,wave_shape", [
    ((2, 3, 12, 20), [12, 12]),  # input width bigger
    ((2, 3, 12, 8), [12, 12])    # input width smaller
])
def test_wavelet2dcwt_size_mismatch(in_shape, wave_shape):
    """
    If input width is bigger/smaller than wavelet_size[-1], 
    wavelet_level is adjusted. No crash, output shape == input shape.
    """
    conv = SpectralConvWavelet2DCwt(
        in_channels=in_shape[1],
        out_channels=2,
        wavelet_level=2,
        wavelet_size=wave_shape,
        wavelet_filter=["near_sym_b", "qshift_b"],
    )
    x = torch.randn(*in_shape)
    y = conv(x)
    assert y.shape == (in_shape[0], 2, in_shape[2], in_shape[3])


def test_wavelet2dcwt_invalid_wavelet_size():
    """
    wavelet_size must be a list of exactly 2 elements.
    """
    with pytest.raises(Exception):
        _ = SpectralConvWavelet2DCwt(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[16],  # Only one dimension
            wavelet_filter=["near_sym_b", "qshift_b"],
        )


@pytest.mark.parametrize(
    "wavelet_filter", [
        ["nonexistent_filter", "qshift_b"],  # invalid filter
        ["near_sym_b", "nonexistent_qshift"] # invalid shift
    ]
)
def test_wavelet2dcwt_invalid_filters(wavelet_filter):
    """
    If wavelet_filter is unrecognized, the underlying DTCWT library 
    typically raises an error.
    """
    with pytest.raises(Exception):
        _ = SpectralConvWavelet2DCwt(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[8, 8],
            wavelet_filter=wavelet_filter,
        )


def test_wavelet2dcwt_weight_shape():
    """
    Check shape of self.weight in dual-tree wavelet 2D:
      shape = (13, in_channels, out_channels, H, W)
    where 13 subbands = 1 approximate + 12 details (6 angles × 2).
    """
    conv = SpectralConvWavelet2DCwt(
        in_channels=2,
        out_channels=3,
        wavelet_level=1,
        wavelet_size=[8, 8],
        wavelet_filter=["near_sym_b", "qshift_b"]
    )
    W = conv.weight
    assert W.dim() == 5, f"Expected 5D weight, got {W.shape}."
    assert W.shape[0] == 13, "Expected 13 subband dimension."
    assert W.shape[1] == 2, "Mismatch in in_channels."
    assert W.shape[2] == 3, "Mismatch in out_channels."
    assert W.shape[3] >= 8 and W.shape[4] >= 8, (
        "The weight's spatial dims should be at least the size of approximate or detail modes."
    )


def test_wavelet2dcwt_mul2d():
    """
    Test the mul2d method in the CWT class:
       mul2d(input, weights) => "bixy, ioxy -> boxy"
    """
    conv = SpectralConvWavelet2DCwt(
        in_channels=2,
        out_channels=2,
        wavelet_level=1,
        wavelet_size=[8, 8],
        wavelet_filter=["near_sym_b", "qshift_b"]
    )
    input_tensor = torch.randn(2, 2, 4, 5)    # (batch=2, in_chan=2, H=4, W=5)
    weight_tensor = torch.randn(2, 2, 4, 5)   # (in_chan=2, out_chan=2, H=4, W=5)
    result = conv.mul2d(input_tensor, weight_tensor)
    assert result.shape == (2, 2, 4, 5)


@pytest.mark.parametrize("wavelet_level", [1, 4])
def test_wavelet2dcwt_edge_levels(wavelet_level):
    """
    Test extremes for wavelet_level in 2D CWT. 
    """
    conv = SpectralConvWavelet2DCwt(
        in_channels=1,
        out_channels=2,
        wavelet_level=wavelet_level,
        wavelet_size=[8, 8],
        wavelet_filter=["near_sym_b", "qshift_b"]
    )
    x = torch.randn(1, 1, 8, 8)
    y = conv(x)
    assert y.shape == (1, 2, 8, 8), f"Got {y.shape} for wavelet_level={wavelet_level}"
    
# -------------------------------------------------------------------------
# TEST SUITE 4: SpectralConvWavelet2DCwt (dual-tree continuous wavelet)
# -------------------------------------------------------------------------
    
@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("wavelet_level", [0, 1, 2, 3])
@pytest.mark.parametrize("wavelet_size", [[8, 8, 8], [16, 8, 12]])
@pytest.mark.parametrize("wavelet_filter", [["db4"], ["haar"]])
@pytest.mark.parametrize("wavelet_mode", ["symmetric", "periodic"])
def test_wavelet3d_forward(
    in_channels,
    out_channels,
    wavelet_level,
    wavelet_size,
    wavelet_filter,
    wavelet_mode
):
    """
    Basic forward test for SpectralConvWavelet3D with various parameters.
    Ensures:
      - No runtime error
      - Output shape matches input shape
      - Output is real-valued if input is real
    """
    batch_size = 2
    D, H, W = wavelet_size
    conv = SpectralConvWavelet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        wavelet_level=wavelet_level,
        wavelet_size=wavelet_size,
        wavelet_filter=wavelet_filter,  # e.g. ['db4']
        wavelet_mode=wavelet_mode,
    )

    x = torch.randn(batch_size, in_channels, D, H, W)
    y = conv(x)
    assert y.shape == (batch_size, out_channels, D, H, W), (
        f"Expected shape {(batch_size, out_channels, D, H, W)}, got {y.shape}"
    )
    assert torch.is_floating_point(y), "Output is expected to be real-valued float Tensor."


@pytest.mark.parametrize("resolution_scaling_factor", [0.5, 2.0, [1.25, 0.5, 2.0]])
def test_wavelet3d_transform(resolution_scaling_factor):
    """
    Test the transform() method in 3D wavelet to ensure up-/down-sampling is correct.
    If resolution_scaling_factor is either a scalar or a list of three, 
    the shape should scale accordingly.
    """
    in_channels, out_channels = 1, 1
    wavelet_size = [8, 8, 8]
    conv = SpectralConvWavelet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        wavelet_level=1,
        wavelet_size=wavelet_size,
        wavelet_filter=["db4"],
        wavelet_mode="periodic",
        resolution_scaling_factor=resolution_scaling_factor
    )
    x = torch.randn(1, in_channels, *wavelet_size)
    x_t = conv.transform(x)

    if isinstance(resolution_scaling_factor, (float, int)):
        expected_shape = tuple(int(s * resolution_scaling_factor) for s in wavelet_size)
    elif isinstance(resolution_scaling_factor, (list, tuple)) and len(resolution_scaling_factor) == 3:
        expected_shape = tuple(int(s * r) for s, r in zip(wavelet_size, resolution_scaling_factor))
    else:
        raise ValueError("Unsupported resolution_scaling_factor format in test.")

    assert x_t.shape[-3:] == expected_shape, (
        f"Expected transform to produce shape {expected_shape}, got {x_t.shape[-3:]}"
    )


@pytest.mark.parametrize("in_shape,wavelet_size", [
    # Input's last dimension bigger than wavelet_size's last dimension
    ((2, 1, 8, 8, 16), [8, 8, 8]),
    # Input's last dimension smaller than wavelet_size's last dimension
    ((2, 1, 8, 8, 4), [8, 8, 8]),
])
def test_wavelet3d_size_mismatch(in_shape, wavelet_size):
    """
    If the input last dimension differs from wavelet_size[-1],
    the code modifies wavelet_level internally. 
    Ensure no crash & the final output shape = input shape.
    """
    batch_size, in_ch, D, H, W = in_shape
    conv = SpectralConvWavelet3D(
        in_channels=in_ch,
        out_channels=2,
        wavelet_level=2,
        wavelet_size=wavelet_size,
        wavelet_filter=["db4"],
        wavelet_mode="periodic",
    )
    x = torch.randn(*in_shape)
    y = conv(x)
    assert y.shape == (batch_size, 2, D, H, W), (
        f"Output shape mismatch. Expected {(batch_size,2,D,H,W)}, got {y.shape}."
    )


def test_wavelet3d_invalid_wavelet_size():
    """
    wavelet_size must be a list of exactly 3 elements for 3D wavelet,
    or the constructor raises an Exception.
    """
    with pytest.raises(Exception, match="WaveConv3d accepts the wavelet_size of 3D signal"):
        _ = SpectralConvWavelet3D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[8, 8],  # Only 2 dims
            wavelet_filter=["db4"],
            wavelet_mode="periodic",
        )

    with pytest.raises(Exception):
        _ = SpectralConvWavelet3D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=(8, 8, 8),  # tuple not a list
            wavelet_filter=["db4"],
            wavelet_mode="periodic",
        )


def test_wavelet3d_weight_shape():
    """
    Check that the internal weight shape matches the expected:
      shape = (8, in_channels, out_channels, modes1, modes2, modes3)
    """
    in_channels, out_channels = 1, 2
    wavelet_size = [8, 8, 8]
    conv = SpectralConvWavelet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        wavelet_level=2,
        wavelet_size=wavelet_size,
        wavelet_filter=["db4"],
        wavelet_mode="periodic"
    )
    W = conv.weight
    assert W.dim() == 6, f"Expected 6D weight, got shape {W.shape}."
    # 8 => approximate (aaa) + 7 detail subbands in 3D (aad, ada, add, daa, dad, dda, ddd)
    assert W.shape[0] == 8, "First dim=8 wavelet subbands for 3D."
    assert W.shape[1] == in_channels, f"Expected in_channels={in_channels}, got {W.shape[1]}"
    assert W.shape[2] == out_channels, f"Expected out_channels={out_channels}, got {W.shape[2]}"
    # modes1, modes2, modes3
    assert W.shape[3] > 0 and W.shape[4] > 0 and W.shape[5] > 0, "Zero dimension in wavelet modes."


def test_wavelet3d_mul3d():
    """
    Direct test of the internal mul3d method:
      mul3d(input, weights) => "ixyz, ioxyz -> oxyz"
      Confirms shape logic is correct.
    """
    conv = SpectralConvWavelet3D(
        in_channels=2,
        out_channels=3,
        wavelet_level=1,
        wavelet_size=[8, 8, 8],
        wavelet_filter=["db4"],
        wavelet_mode="periodic"
    )
    # input: shape (in_channels=2, D=4, H=5, W=6)
    inp = torch.randn(2, 4, 5, 6)
    # weights: shape (in_channels=2, out_channels=3, D=4, H=5, W=6)
    wts = torch.randn(2, 3, 4, 5, 6)
    out = conv.mul3d(inp, wts)
    assert out.shape == (3, 4, 5, 6), f"Got shape {out.shape}."


@pytest.mark.parametrize("wavelet_filter", [["db2"], ["haar"], ["coif1"]])
def test_wavelet3d_forward_different_filters(wavelet_filter):
    """
    Check that recognized wavelet filters in pywt do not crash in 3D usage.
    """
    conv = SpectralConvWavelet3D(
        in_channels=1,
        out_channels=1,
        wavelet_level=2,
        wavelet_size=[8, 8, 8],
        wavelet_filter=wavelet_filter,
        wavelet_mode="periodic"
    )
    x = torch.randn(1, 1, 8, 8, 8)
    y = conv(x)
    assert y.shape == (1, 1, 8, 8, 8)


def test_wavelet3d_unrecognized_filter():
    """
    If user specifies an unknown wavelet_filter, 
    we expect pywt or wavedec3 to raise an error.
    """
    with pytest.raises(Exception):
        _ = SpectralConvWavelet3D(
            in_channels=1,
            out_channels=1,
            wavelet_level=1,
            wavelet_size=[8, 8, 8],
            wavelet_filter=["nonexistent_wavelet"],
            wavelet_mode="periodic",
        )