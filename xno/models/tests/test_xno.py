from math import prod

import pytest
import torch
from tensorly import tenalg
from configmypy import Bunch

from xno.models import XNO, XNO1d, XNO2d, XNO3d

tenalg.set_backend("einsum")


# @pytest.mark.parametrize(
#     "factorization", ["ComplexDense", "ComplexTucker", "ComplexCP", "ComplexTT"]
# )
# @pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("fno_block_precision", ["full", "half", "mixed"])
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("lifting_channel_ratio", [1, 2])
@pytest.mark.parametrize("preactivation", [False, True])
def test_tfno(
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channel_ratio,
    preactivation,
):
    if torch.has_cuda:
        device = "cuda"
        s = 16
        modes = 8
        width = 16
        fc_channels = 16
        batch_size = 4
        n_layers = 4
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 16
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2

    dtype = torch.float 
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = XNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        rank=rank,
        fixed_rank_modes=False,
        n_layers=n_layers,
        stabilizer=stabilizer,
        fc_channels=fc_channels,
        lifting_channel_ratio=lifting_channel_ratio,
        preactivation=preactivation,
        fno_block_precision=fno_block_precision
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size, dtype=dtype).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    # take the modulus if data is complex-valued to create grad
    if dtype == torch.cfloat:
        loss = (loss.real ** 2 + loss.imag ** 2) ** 0.5
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"


@pytest.mark.parametrize(
    "resolution_scaling_factor",
    [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [1, 2, 2],
        [1, 0.5, 1],
    ],
)
def test_fno_superresolution(resolution_scaling_factor):
    device = "cpu"
    s = 16
    modes = 5
    hidden_channels = 15
    fc_channels = 32
    batch_size = 3
    n_layers = 3
    n_dim = 2
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = XNO(
        in_channels=3,
        out_channels=1,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        factorization="cp",
        implementation="reconstructed",
        rank=rank,
        resolution_scaling_factor=resolution_scaling_factor,
        n_layers=n_layers,
        fc_channels=fc_channels,
    ).to(device)

    print(f"{model.resolution_scaling_factor=}")

    in_data = torch.randn(batch_size, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(resolution_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]


# ======================================================
# ================== Extra test cases ==================
# ======================================================


# test_xno.py

import pytest
import torch
import torch.nn.functional as F

# Adjust these imports to your actual project/module paths:
# from xno.models.xno import XNO, XNO1d, XNO2d, XNO3d, TXNO, TXNO1d, TXNO2d, TXNO3d

try:
    import pywt  # If wavelet-based transformations might be tested
except ImportError:
    pass

@pytest.fixture(params=["FNO", "HNO", "LNO", "WNO"])
def transformation(request):
    return request.param

@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_XNO_forward_basic(n_dim, transformation):
    """
    Basic forward test for XNO with varying dimensionality.
    Ensures:
      - No runtime error
      - Output shape matches input shape except for channel dimension
    """
    
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:n_dim],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
    # Suppose we have n_modes=4 for each dimension
    n_modes = (4,) * n_dim
    in_channels, out_channels, hidden_channels = 3, 5, 8
    xno = XNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        transformation=transformation,      # minimal scenario
        n_layers=2,
        transformation_kwargs=transformation_kwargs
    )

    # Construct an input: shape (batch=2, in_channels, *spatial)
    size = [10] * n_dim
    x = torch.randn(2, in_channels, *size)

    y = xno(x)
    assert y.shape == (2, out_channels, *size), (
        f"Expected shape (2, {out_channels}, {size}), got {y.shape}"
    )


@pytest.mark.parametrize("embedding_option", [None, "grid"])
def test_XNO_positional_embedding(embedding_option, transformation):
    """
    Tests XNO with or without a positional embedding:
    - If 'grid', the embedding adds 'n_dim' channels to the input
    - If None, no extra channels are added
    """
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
    xno = XNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        positional_embedding=embedding_option,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs
    )
    x = torch.randn(1, 2, 10, 10)
    y = xno(x)
    assert y.shape == (1, 2, 10, 10)

def test_XNO_custom_embedding(transformation):
    """
    If user passes a custom GridEmbedding2D or GridEmbeddingND,
    we ensure the XNO constructor does not fail and forward pass is correct.
    """
    from xno.layers.embeddings import GridEmbeddingND

    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
    embedding = GridEmbeddingND(in_channels=2, dim=2, grid_boundaries=[[0., 1.], [0., 1.]])
    xno = XNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        positional_embedding=embedding,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs
    )
    x = torch.randn(1, 2, 16, 16)
    y = xno(x)
    assert y.shape == (1, 2, 16, 16)


@pytest.mark.parametrize("domain_padding_value", [None, 0.1, [0.2, 0.3]])
def test_XNO_domain_padding(domain_padding_value, transformation):
    """
    Tests domain padding logic. If domain_padding is not None and > 0,
    the input is padded then unpadded internally. The final output shape 
    must match the original (batch, out_channels, *spatial).
    """
    n_dim = 2
    n_modes = (4, 4)
    in_channels, out_channels, hidden_channels = 1, 1, 4
    
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:n_dim],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
    xno = XNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        domain_padding=domain_padding_value,
        domain_padding_mode="one-sided",  # or "symmetric"
        transformation=transformation,
        transformation_kwargs=transformation_kwargs
    )

    size = [8] * n_dim
    x = torch.randn(1, in_channels, *size)
    y = xno(x)
    assert y.shape == (1, out_channels, *size), (
        f"Expected shape (1, {out_channels}, {size}), got {y.shape}"
    )


@pytest.mark.parametrize("scaling_factor", [0.5, 2.0, [1.25, 0.75]])
def test_XNO_resolution_scaling_factor(scaling_factor, transformation):
    """
    Tests that we can specify resolution_scaling_factor in the XNO constructor.
    The up/down sampling is applied in each XNOBlock's forward pass.
    """
    n_dim = 2
    
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
        
    xno = XNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=3,
        hidden_channels=4,
        resolution_scaling_factor=scaling_factor,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs,
        n_layers=2,
    )
    size = [8] * n_dim
    x = torch.randn(1, 2, *size)
    y = xno(x)
    # The final shape after the last XNOBlock's transform
    # if scaling_factor is a single float, or a list for each layer.
    # We won't do an exact shape check unless we replicate the logic.
    # At minimum, ensure no crash and out_channels match.
    assert y.shape[1] == 3, f"Expected out_channels=3, got {y.shape[1]}"

def test_XNO_n_modes_property(transformation):
    """
    Check that updating xno.n_modes also updates the underlying XNOBlocks.
    """
    
    n_dim = 2
    
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
    xno = XNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs,
        n_layers=2
    )
    # Initially (4,4)
    assert tuple(xno.xno_blocks.n_modes) == (4, 4)
    # Update n_modes property
    xno.n_modes = (6, 6)
    assert tuple(xno.xno_blocks.n_modes) == (6, 6)


def test_XNO_transformation_kwargs(transformation):
    """
    If the transformation is WNO or LNO, we might pass 'transformation_kwargs'.
    The main test is that the constructor doesn't crash with minimal/valid
    arguments. 
    """
    if transformation.lower() == "wno":
        # Example wavelet args
        tf_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": [8, 8],
            "wavelet_filter": ["db2"],
            "wavelet_mode": "symmetric"
        }
    elif transformation.lower() == "lno":
        # Possibly no special transformation_kwargs needed
        tf_kwargs = {}
    else:
        tf_kwargs = {}

    xno = XNO(
        n_modes=(4, 4),
        in_channels=1,
        out_channels=1,
        hidden_channels=4,
        transformation=transformation,
        transformation_kwargs=tf_kwargs,
        n_layers=1
    )
    x = torch.randn(1, 1, 8, 8)
    y = xno(x)
    assert y.shape == (1, 1, 8, 8)


def test_XNO_call_with_output_shape(transformation):
    """
    Tests that 'output_shape' can be passed to forward() to override the final shape 
    in the last XNOBlock. 
    Usually used if the user wants to resize back to some shape at the last layer.
    """
    
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
        
    xno = XNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs,
        n_layers=2
    )
    x = torch.randn(1, 2, 16, 16)
    # Suppose we want the final output to be (16, 8) in spatial dims
    out_sh = (16, 8)
    y = xno(x, output_shape=out_sh)
    assert y.shape == (1, 2, 16, 8)


def test_XNO1d_forward(transformation):
    """
    Test the 1D specialized XNO1d class.
    """
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:1],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
        
    model = XNO1d(
        n_modes_height=4,
        hidden_channels=8,
        in_channels=2,
        out_channels=2,
        n_layers=2,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs,
    )
    x = torch.randn(1, 2, 16)  # 1D => shape: (batch=1, in_channels=2, length=16)
    y = model(x)
    assert y.shape == (1, 2, 16)


def test_XNO2d_forward(transformation):
    """
    Test the 2D specialized XNO2d class.
    """
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:2],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
        
    model = XNO2d(
        n_modes_height=4,
        n_modes_width=4,
        hidden_channels=8,
        in_channels=2,
        out_channels=3,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs,
        n_layers=1
    )
    x = torch.randn(2, 2, 12, 12)
    y = model(x)
    assert y.shape == (2, 3, 12, 12)


def test_XNO3d_forward(transformation):
    """
    Test the 3D specialized XNO3d class.
    """
    transformation_kwargs = {}
    if transformation == "WNO":
        wavelet_size = [2, 2, 2]
        transformation_kwargs = {
            "wavelet_level": 1,
            "wavelet_size": wavelet_size[:3],
            "wavelet_filter": ["db4"],  # or your wavelet filter
            "wavelet_mode": "symmetric"
        }
        
    model = XNO3d(
        n_modes_height=4,
        n_modes_width=4,
        n_modes_depth=2,
        hidden_channels=8,
        in_channels=1,
        out_channels=1,
        transformation=transformation,
        transformation_kwargs=transformation_kwargs,
        n_layers=1
    )
    x = torch.randn(1, 1, 6, 6, 4)
    y = model(x)
    assert y.shape == (1, 1, 6, 6, 4)


def test_TXNO_factorization():
    """
    Tests the partialclass 'TXNO' which sets factorization='Tucker'.
    Ensures no crash and shape correctness for the forward pass.
    """
    from xno.models import TXNO  # or from ..xno import TXNO

    model = TXNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        n_layers=1
    )
    x = torch.randn(1, 2, 16, 16)
    y = model(x)
    assert y.shape == (1, 2, 16, 16), (
        f"Expected (1,2,16,16), got {y.shape}"
    )
