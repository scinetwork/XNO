import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Sequence

class SpectralConvND(nn.Module):
    """
    A generalized N-dimensional spectral convolution layer compatible with the Neural Operator framework.
    This layer performs a Fourier transform of the input, applies a learned filter (in spectral space),
    and transforms it back. It is analogous to the SpectralConv layers used by FNO, but generalized
    to an arbitrary number of dimensions.

    Assumptions and Approach:
    - Input shape: (batch, in_channels, d_1, d_2, ..., d_n)
    - Output shape: (batch, out_channels, d_1, d_2, ..., d_n)
    - The number of spatial/temporal dimensions n is inferred from the input tensor shape.
    - "modes" can be an integer or a sequence. If an integer, the same number of modes is used in all dimensions.
    - Domain lengths for each dimension can be provided; if not, a default length of 1.0 is assumed for each dimension.
      From these lengths and input 2sizes, we compute uniform spacing (dt) along each axis.
    - Frequencies are computed using torch.fft.fftfreq based on spacing.
    - The layer learns complex-valued spectral weights for each frequency mode.
    - Truncation to modes is applied symmetrically around zero frequency. Frequencies beyond "modes_i" are not modeled.

    This class focuses on a clear, extensible design that can be easily integrated into larger operator networks.

    Example:
        conv = SpectralConvND(in_channels=4, out_channels=8, modes=8, domain=[1.0, 1.0, 0.5])
        x = torch.randn(10, 4, 64, 64, 32)  # batch=10, in_ch=4, dimensions: 64x64x32
        y = conv(x)  # y will be [10, 8, 64, 64, 32]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        domain: Optional[Sequence[float]] = None
    ):
        super(SpectralConvND, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # If modes is an integer, use the same number of modes in all dimensions
        if isinstance(modes, int):
            self.modes = [modes]  # Will be expanded once input is known
        else:
            self.modes = list(modes)

        # Domain lengths per dimension. If not provided, assume 1.0 for each dimension.
        # The actual number of dimensions will be inferred from the input.
        self.domain = domain  # List of floats or None
        # We will finalize domain lengths once we know the input shape in forward.

        # Weights will be initialized after knowing the dimension count
        self.weights = None

    def _initialize_weights(self, dim: int):
        """
        Initialize the spectral weights after we know how many dimensions we have.

        Parameters:
        dim (int): Number of spatial/temporal dimensions (excluding batch and channels).
        """
        # If self.modes was only partially defined and dim > len(self.modes),
        # we assume the same modes for all dimensions.
        if len(self.modes) == 1 and dim > 1:
            self.modes = self.modes * dim
        elif len(self.modes) != dim:
            raise ValueError(f"Number of modes provided ({len(self.modes)}) does not match input dimensionality ({dim}).")

        # Initialize spectral weights:
        # Shape: [in_channels, out_channels, modes_1, modes_2, ..., modes_dim]
        # Complex weights: real + imaginary parts
        weight_shape = [self.in_channels, self.out_channels] + self.modes
        self.weights = nn.Parameter(
            torch.rand(weight_shape, dtype=torch.cfloat) * (1 / (self.in_channels * self.out_channels))
        )

    def _compute_dt(self, shape: Sequence[int]) -> Sequence[float]:
        """
        Compute uniform spacing (dt) for each dimension based on domain lengths and shape.
        If domain is None, assume each dimension length = 1.0.

        Parameters:
        shape (Sequence[int]): The shape of the input excluding batch and channel, i.e. (d_1, d_2, ..., d_n).

        Returns:
        dt_list (Sequence[float]): A list of spacings, one per dimension.
        """
        dim = len(shape)
        if self.domain is None:
            domain_lengths = [1.0] * dim
        else:
            if len(self.domain) != dim:
                raise ValueError(f"Domain length ({len(self.domain)}) does not match number of input dimensions ({dim}).")
            domain_lengths = self.domain

        # Uniform spacing: length / n_points
        dt_list = [domain_lengths[i] / shape[i] for i in range(dim)]
        return dt_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spectral convolution.

        Steps:
        1. Infer number of dimensions n from input shape.
        2. If weights not initialized, initialize them.
        3. Compute dt for each dimension.
        4. Compute FFT of input.
        5. Truncate frequencies to the specified modes and multiply by learned spectral weights.
        6. Reconstruct output by inverse FFT.
        """

        # x shape: (batch, in_channels, d_1, d_2, ..., d_n)
        bsz, in_ch = x.size(0), x.size(1)
        spatial_shape = x.shape[2:]
        dim = len(spatial_shape)

        if self.weights is None:
            self._initialize_weights(dim)

        # Compute dt for each dimension
        dt_list = self._compute_dt(spatial_shape)

        # Compute frequencies for each dimension to understand the spectral domain:
        # Although for standard FNO-like approach we might not need frequencies explicitly,
        # let's keep them for clarity. Frequencies could be useful for advanced modifications.
        freqs = []
        for size, dt in zip(spatial_shape, dt_list):
            freq = torch.fft.fftfreq(size, d=dt).to(x.device)
            freqs.append(freq)
        # freqs is a list of frequency tensors for each dimension

        # Compute n-dimensional FFT
        # FFT along last `dim` axes
        dims = tuple(range(2, 2 + dim))
        x_ft = torch.fft.fftn(x, dim=dims)

        # Truncate high frequency modes
        # We assume symmetric truncation. For simplicity, we only keep the first 'modes[i]' frequencies
        # along each dimension. Typically, FNO truncation keeps modes from 0 to modes_i (the low frequencies).
        # Here we assume the input is complex and we handle only the positive modes for demonstration.
        # If negative frequencies are needed, you would have to carefully select symmetric indices.
        # For simplicity, we only keep low-frequency slices starting at zero frequency.
        
        slices = [slice(None), slice(None)]  # for batch and in_channels
        for m, dsz in zip(self.modes, spatial_shape):
            # m should not exceed half of the dimension size if symmetrical truncation is desired.
            # We'll just take from 0 up to m. (For a more physically correct truncation, consider negative freqs)
            slices.append(slice(0, min(m, dsz)))
        
        # x_ft truncated
        x_ft_low = x_ft[tuple(slices)]  # shape: (batch, in_channels, *modes...)

        # Multiply by the spectral weights:
        # weights: [in_channels, out_channels, *modes...]
        # x_ft_low: [batch, in_channels, *modes...]
        # Einsum pattern: (b, i, m1, m2, ..., mn) * (i, o, m1, m2, ..., mn) -> (b, o, m1, m2, ..., mn)
        # We'll use torch.einsum for clarity:
        # eq = 'bim1m2...mn, iom1m2...mn -> bom1m2...mn'
        # Construct einsum string dynamically:
        indices = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # For n dimensions: input has b,i plus n dims; weight has i,o plus n dims.
        # Example for n=2: x_ft_low: b i m1 m2, weight: i o m1 m2 -> b o m1 m2
        # For general n: x_ft_low: b i m1 ... mn, weight: i o m1 ... mn -> b o m1 ... mn
        in_str = "b" + "i" + "".join(indices[:dim])
        w_str = "i" + "o" + "".join(indices[:dim])
        out_str = "b" + "o" + "".join(indices[:dim])
        eq = f'{in_str},{w_str}->{out_str}'

        y_ft_low = torch.einsum(eq, x_ft_low, self.weights)

        # Now we have truncated output in Fourier space. We need to put it back into a full-size tensor
        # before inverse transforming. We'll reconstruct a full frequency tensor (zero-pad high frequencies).
        # Start from a zeros tensor of full spatial size:
        y_ft_full = torch.zeros(
            bsz, self.out_channels, *spatial_shape,
            dtype=y_ft_low.dtype,
            device=y_ft_low.device
        )

        # Insert the low frequency components back
        insert_slices = [slice(None), slice(None)]
        for m, dsz in zip(self.modes, spatial_shape):
            insert_slices.append(slice(0, min(m, dsz)))
        y_ft_full[tuple(insert_slices)] = y_ft_low

        # Inverse FFT to get back to spatial domain
        y = torch.fft.ifftn(y_ft_full, dim=dims)
        # Typically, the final output of FNO is real. If we want a real result:
        y = y.real

        return y
