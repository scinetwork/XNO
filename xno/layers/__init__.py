# MIT License
# Copyright (c) 2024 Saman Pordanesh
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software...

from .spectral_convolution_fourier import SpectralConvFourier
from .spectral_convolution_hilbert import SpectralConvHilbert
from .spectral_convolution_laplace import SpectralConvLaplace1D, SpectralConvLaplace2D, SpectralConvLaplace3D
from .spectral_convolution_wavelet import SpectralConvWavelet1D, SpectralConvWavelet2D, SpectralConvWavelet3D

# Map transformation types to their corresponding convolution classes
CONVOLUTION_MAP = {
    "fno": SpectralConvFourier,
    "hno": SpectralConvHilbert,
    "lno": {
        1: SpectralConvLaplace1D,
        2: SpectralConvLaplace2D,
        3: SpectralConvLaplace3D,
    },
    "wno": {
        1: SpectralConvWavelet1D,
        2: SpectralConvWavelet2D,
        3: SpectralConvWavelet3D,
    },
}

def get_spectral_convolution_class(transformation: str, dim: int):
    """
    Retrieve the appropriate spectral convolution class based on the transformation
    type and dimensionality.

    Parameters
    ----------
    transformation : str
        The transformation type (e.g., "fno", "hno", "lno", "wno").
    dim : int
        The number of dimensions (e.g., 1D, 2D, 3D).

    Returns
    -------
    Class
        The spectral convolution class corresponding to the transformation and dim.

    Raises
    ------
    ValueError
        If the transformation type or dimensionality is not supported.
    """
    transformation = transformation.lower()
    if transformation not in CONVOLUTION_MAP:
        raise ValueError(f"Unknown transformation '{transformation}'. Supported: {list(CONVOLUTION_MAP.keys())}")
    
    conv_entry = CONVOLUTION_MAP[transformation]
    
    if isinstance(conv_entry, dict):
        # Handle dimensionality-specific transformations (e.g., LNO, WNO)
        if dim not in conv_entry:
            raise ValueError(f"Unsupported dimension '{dim}' for transformation '{transformation}'.")
        return conv_entry[dim]
    
    # For non-dimensional-specific transformations (e.g., FNO, HNO)
    return conv_entry
