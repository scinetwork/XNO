# test_xyno.py

import pytest
import torch
import torch.nn.functional as F

from xno.models.xyno import XYNO, XYNO1d, XYNO2d, XYNO3d, TXYNO, TXYNO1d, TXYNO2d, TXYNO3d

###############################################################################
# Helpers
###############################################################################

def wavelet_kwargs_for_dim(dim):
    """
    Returns a minimal dictionary of wavelet parameters matching `dim`.
    For example, if dim=2 => wavelet_size is 2D, etc.
    """
    # This is just an example. Adjust 'wavelet_size' or 'wavelet_level' to your liking.
    base_size = [8, 8, 4]  # covers up to 3D
    return {
        "wavelet_level": 1,
        "wavelet_size": base_size[:dim],
        "wavelet_filter": ["db4"],
        "wavelet_mode": "symmetric",
    }

def random_input(batch_size, in_channels, shape, complex_data=False):
    """Creates a random input tensor with optional complex dtype."""
    if complex_data:
        return torch.randn(*((batch_size, in_channels) + shape), dtype=torch.cfloat)
    else:
        return torch.randn(*((batch_size, in_channels) + shape))


###############################################################################
# 1) Test that we can build XYNO with both mix_mode='parallel' and 'pure'
#    with different kernel sets, for various dimension sizes.
###############################################################################

@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("mix_mode", ["parallel", "pure"])
@pytest.mark.parametrize("parallel_kernels", [
    ["fno"],                     # single kernel
    ["fno", "lno"],             # multiple kernels in parallel
    ["fno", "hno", "wno", "lno"]  # mix of all
])
@pytest.mark.parametrize("pure_kernels_order", [
    ["fno"],                       # single kernel => implies 1 layer
    ["fno", "wno"],               # 2 layers
    ["lno", "hno", "fno"]         # 3 layers
])
def test_XYNO_basic_build_and_forward(
    n_dim, mix_mode, parallel_kernels, pure_kernels_order
):
    """
    Checks that XYNO can:
      1) Build with no immediate errors
      2) Forward pass returns correct shape
    """
    # We'll set up a scenario:
    #   - If mix_mode='parallel', we'll ignore pure_kernels_order
    #   - If mix_mode='pure', the # of layers is len(pure_kernels_order)
    #   - If 'wno' is in the chosen kernels, we pass wavelet_kwargs

    # Decide the dimension size and wavelet parameters if needed
    shape = (8,) * n_dim
    # Build a minimal transformation_kwargs if wno is among kernels:
    need_wavelet = ("wno" in parallel_kernels) or ("wno" in pure_kernels_order)
    transformation_kwargs = wavelet_kwargs_for_dim(n_dim) if need_wavelet else None

    # Decide #layers
    n_layers = 4  # default
    if mix_mode == "pure":
        # override n_layers to match pure_kernels_order length
        n_layers = len(pure_kernels_order)

    # Construct the model
    model = XYNO(
        n_modes=(4,)*n_dim,
        in_channels=2,
        out_channels=3,
        hidden_channels=5,
        mix_mode=mix_mode,
        parallel_kernels=parallel_kernels,
        pure_kernels_order=pure_kernels_order,
        transformation_kwargs=transformation_kwargs,
        n_layers=n_layers,
    )

    # Build random input
    x = random_input(batch_size=2, in_channels=2, shape=shape)

    # Forward
    y = model(x)
    assert y.shape[:2] == (2, 3), f"Expected batch=2, out_channels=3, got {y.shape[:2]}"
    assert y.shape[2:] == shape, f"Expected spatial shape {shape}, got {y.shape[2:]}"


###############################################################################
# 2) Test that unknown kernel in parallel_kernels or pure_kernels_order raises ValueError
###############################################################################

@pytest.mark.parametrize("mix_mode", ["parallel", "pure"])
def test_XYNO_unknown_kernel(mix_mode):
    with pytest.raises(ValueError):
        if mix_mode == "parallel":
            XYNO(
                n_modes=(4, 4),
                in_channels=2,
                out_channels=2,
                hidden_channels=4,
                mix_mode=mix_mode,
                parallel_kernels=["fno", "unknown_kernel"],  # error
            )
        else:
            XYNO(
                n_modes=(4, 4),
                in_channels=2,
                out_channels=2,
                hidden_channels=4,
                mix_mode=mix_mode,
                pure_kernels_order=["fno", "hno", "unknown_kernel"],  # error
            )

###############################################################################
# 3) Test that WNO requires wavelet_kwargs => missing => ValueError
###############################################################################

def test_XYNO_missing_wavelet_kwargs():
    # We'll set mix_mode='parallel' with 'wno' in parallel_kernels
    with pytest.raises(ValueError):
        XYNO(
            n_modes=(4, 4),
            in_channels=2,
            out_channels=2,
            hidden_channels=4,
            mix_mode="parallel",
            parallel_kernels=["wno"],   # requires wavelet params
            transformation_kwargs=None, # missing => error
        )


###############################################################################
# 4) Test domain padding logic
###############################################################################

@pytest.mark.parametrize("domain_padding_value", [None, 0.1, [0.2, 0.3]])
def test_XYNO_domain_padding(domain_padding_value):
    # 2D, parallel mode for simplicity
    model = XYNO(
        n_modes=(4, 4),
        in_channels=1,
        out_channels=1,
        hidden_channels=4,
        mix_mode="parallel",
        parallel_kernels=["fno"],
        domain_padding=domain_padding_value,
        domain_padding_mode="one-sided"
    )
    x = torch.randn(1, 1, 8, 8)
    y = model(x)
    assert y.shape == (1, 1, 8, 8), "Domain padding must unpad back to original shape."


###############################################################################
# 5) Test resolution scaling factor
###############################################################################

@pytest.mark.parametrize("scaling_factor", [1.5, 2.0, [1.25, 1.75]])
@pytest.mark.parametrize("mix_mode", ["parallel", "pure"])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_XYNO_resolution_scaling(mix_mode, scaling_factor, n_dim):
    # n_dim = 2
    # If pure => n_layers=2, if parallel => keep n_layers=2 for clarity
    pure_kernels_order = ["fno", "wno"]  # just 2 layers
    parallel_kernels = ["fno", "wno"]    # in parallel
    shape = (8,)*n_dim
    n_modes = (4,)*n_dim
    model = XYNO(
        n_modes=n_modes,
        in_channels=1,
        out_channels=2,
        hidden_channels=3,
        mix_mode=mix_mode,
        parallel_kernels=parallel_kernels,
        pure_kernels_order=pure_kernels_order,
        n_layers=2,  # matches pure_kernels length if mix_mode='pure'
        transformation_kwargs=wavelet_kwargs_for_dim(n_dim),
        resolution_scaling_factor=scaling_factor,
    )
    x = torch.randn(1, 1, *shape)
    y = model(x)
    # We won't do an exact shape match, but ensure no crash and channels=2
    assert y.shape[1] == 2

###############################################################################
# 6) Test positional embedding
###############################################################################

@pytest.mark.parametrize("positional_embedding", [None, "grid"])
def test_XYNO_positional_embedding(positional_embedding):
    model = XYNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        mix_mode="parallel",
        parallel_kernels=["fno"],
        positional_embedding=positional_embedding
    )
    x = torch.randn(1, 2, 8, 8)
    y = model(x)
    assert y.shape == (1, 2, 8, 8)


def test_XYNO_custom_embedding():
    """
    If user passes a custom embedding module like GridEmbeddingND, 
    verify that it runs and shape is correct.
    """
    # Adjust import path to your project
    from xno.layers.embeddings import GridEmbeddingND
    embedding = GridEmbeddingND(in_channels=2, dim=2)
    
    model = XYNO(
        n_modes=(4,4),
        in_channels=2,
        out_channels=3,
        hidden_channels=5,
        mix_mode="parallel",
        parallel_kernels=["fno"],
        positional_embedding=embedding,
    )
    x = torch.randn(1, 2, 8, 8)
    y = model(x)
    assert y.shape == (1, 3, 8, 8)


###############################################################################
# 7) Test n_modes property setter
###############################################################################

def test_XYNO_n_modes_property():
    model = XYNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        mix_mode="parallel",
        parallel_kernels=["fno"]
    )
    assert tuple(model.xno_blocks.n_modes) == (4,4)
    # Now update
    model.n_modes = (6, 6)
    assert tuple(model.xno_blocks.n_modes) == (6,6)


###############################################################################
# 8) Test forward(...) with explicit 'output_shape'
###############################################################################

@pytest.mark.parametrize("mix_mode", ["parallel", "pure"])
def test_XYNO_output_shape_override(mix_mode):
    n_dim = 2
    shape = (8, 8)
    n_layers = 2
    if mix_mode == "pure":
        # pure_kernels_order must match n_layers
        pure_kernels_order = ["fno", "hno"]
    else:
        pure_kernels_order = ["fno", "fno"]  # won't matter in parallel

    model = XYNO(
        n_modes=(4,4),
        in_channels=1,
        out_channels=2,
        hidden_channels=3,
        mix_mode=mix_mode,
        parallel_kernels=["fno", "hno"],
        pure_kernels_order=pure_kernels_order,
        n_layers=n_layers
    )
    x = torch.randn(1, 1, *shape)
    # Suppose we want the final shape to be (4, 8) in spatial dims
    out_sh = (4, 8)
    y = model(x, output_shape=out_sh)
    assert y.shape == (1, 2, 4, 8)

###############################################################################
# 9) Test partial classes (TXYNO, TXYNO1d, etc.)
###############################################################################

def test_TXYNO():
    # Suppose factorization='Tucker' is set by partialclass
    from xno.models.xyno import TXYNO  # adjust import
    model = TXYNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        hidden_channels=4,
        mix_mode="parallel",
        parallel_kernels=["fno"],
        n_layers=1
    )
    x = torch.randn(1, 2, 8, 8)
    y = model(x)
    assert y.shape == (1, 2, 8, 8)
