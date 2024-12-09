import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

Number = Union[int, float]


# Example imports (adjust paths according to your file structure)
from .spectral_convolution_hilbert import SpectralConvHilbert
from .spectral_convolution_fourier import SpectralConvFourier

class SpectralConv(nn.Module):
    """
    A wrapper class that selects a specific spectral convolution implementation
    (e.g., Fourier, Hilbert, Laplace) based on the 'transform' argument.
    
    Parameters
    ----------
    transform : str
        The type of spectral transform to use. Must be one of 'fourier', 'hilbert', 'laplace'.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    **kwargs : dict
        Additional keyword arguments passed to the underlying spectral convolution class.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        transform="fno", 
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        ):
        super().__init__()
        
        # Map transform types to their respective classes
        SC_MAP = {
            'fno': SpectralConvFourier,
            'hno': SpectralConvHilbert,
        }
        
        transform = transform.lower()
        if transform not in SC_MAP:
            raise ValueError(
                f"Unknown transform type '{transform}'. "
                f"Available transforms: {list(SC_MAP.keys())}"
            )
        
        # Instantiate the selected spectral convolution module
        self.sc = SC_MAP[transform](
                                    in_channels,
                                    out_channels,
                                    n_modes,
                                    complex_data,
                                    max_n_modes,
                                    bias,
                                    separable, 
                                    resolution_scaling_factor,
                                    fno_block_precision,
                                    rank,
                                    factorization,
                                    implementation,
                                    fixed_rank_modes,
                                    decomposition_kwargs,
                                    init_std,
                                    fft_norm,
                                    device,
                                    **kwargs
                                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that delegates to the selected spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_channels, ...] (dimensions depend on the operator).

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, out_channels, ...].
        """
        return self.sc(x)
