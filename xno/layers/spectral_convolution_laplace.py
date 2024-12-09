from typing import List, Optional, Tuple, Union

from ..utils import validate_scaling_factor

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .resample import resample

import numpy as np

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    
    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors. 
        if False, 
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]


class SpectralConvLaplace(BaseSpectralConv):
    """SpectralConvLaplace implements a Laplace-domain spectral convolution 
    using the pole-residue method, generalized for nD.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or list[int]
        Number of modes along each dimension.
    complex_data : bool, optional
        Whether input data is complex, by default False
    max_n_modes : list[int] or None, optional
        Max modes (similar to SpectralConvFourier), by default None
    bias : bool, optional
        If True, learns a bias term, by default True
    separable : bool, optional
        Whether to use separable weights, by default False
    resolution_scaling_factor : float or list[float], optional
        Resolution scaling, by default None
    xno_block_precision : str, optional
        Precision setting ('full', 'half', 'mixed'), by default 'full'
    rank : float, optional
        Factorization rank, by default 0.5
    factorization : str or None
        Factorization type ('tucker','cp','tt') or None
    implementation : {'factorized', 'reconstructed'}
        If factorization is not None, sets forward method
    fixed_rank_modes : bool or list[int]
        Whether certain modes are kept fixed rank
    decomposition_kwargs : dict, optional
        Extra kwargs for factorization
    init_std : float or 'auto', optional
        Init std for weights
    fft_norm : str, optional
        FFT normalization, by default 'forward'

    Notes
    -----
    This class mimics the structure and flexibility of SpectralConvFourier, but 
    applies a Laplace transform approach using poles and residues:
    alpha = FFT(x)
    Compute lambda from fftfreq
    Hw = weights_residue * product over i (1/(lambda_i - weights_pole_i))
    output_residue1,2 = contraction of alpha with Hw
    inverse FFT for transient part
    exponentials for steady-state part.

    Minimal changes outside the class: The same contracting logic and factorization 
    helpers are used.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        xno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_data = complex_data

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.order = len(n_modes)
        self._n_modes = None
        self.n_modes = n_modes

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.xno_block_precision = xno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.separable = separable
        self.fft_norm = fft_norm

        self.resolution_scaling_factor = validate_scaling_factor(resolution_scaling_factor, self.order)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        # For Laplace we have poles and residues.
        # We'll store one weights_residue tensor of shape (in_channels, out_channels, *max_n_modes)
        # and a list of weights_pole tensors of shape (in_channels, out_channels, max_n_modes[i]) for each dimension i.
        # This generalizes the 1D/2D/3D approach.

        factorization = factorization if factorization is not None else "Dense"

        residue_shape = (in_channels, out_channels, *max_n_modes)
        if self.separable:
            if in_channels != out_channels:
                raise ValueError("Separable requires in_channels=out_channels")
            residue_shape = (in_channels, *max_n_modes)

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        # Create weights_residue
        self.weights_residue = FactorizedTensor.new(
            residue_shape, rank=self.rank, factorization=factorization,
            fixed_rank_modes=fixed_rank_modes, dtype=torch.cfloat, **tensor_kwargs
        )
        self.weights_residue.normal_(0, init_std)

        # Create weights_pole for each dimension
        # Poles are 1D along each dimension's modes, but must still consider in/out channels.
        # Similar to how Fourier has a single weight, we have a factorized tensor for poles as well.
        # We'll store a list for clarity. Each is (in_channels, out_channels, max_n_modes[i]).
        self.weights_pole = nn.ParameterList()
        for i, m in enumerate(max_n_modes):
            pole_shape = (in_channels, out_channels, m)
            if self.separable:
                pole_shape = (in_channels, m)
            if factorization.lower() == "dense":
                w_pole = nn.Parameter(init_std * torch.randn(*pole_shape, dtype=torch.cfloat))
            else:
                # factorize poles as well
                w_pole = FactorizedTensor.new(
                    pole_shape, rank=self.rank, factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes, dtype=torch.cfloat, **tensor_kwargs
                )
                w_pole.normal_(0, init_std)
            self.weights_pole.append(w_pole)

        self._contract_residues = get_contract_fun(self.weights_residue, implementation=implementation, separable=self.separable)
        # For poles, we won't contract them the same way (they are used differently),
        # but we might need a helper to get them as tensors if factorized:
        def get_pole_as_tensor(p):
            if isinstance(p, FactorizedTensor):
                return p.to_tensor()
            return p

        self.get_poles = lambda: [get_pole_as_tensor(p) for p in self.weights_pole]

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        n_modes = list(n_modes)
        # Adjust last mode for real-valued spatial data if needed
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])
        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor))
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape
        if in_shape == list(out_shape):
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None, grid: Optional[List[torch.Tensor]]=None):
        """Forward pass for Laplace spectral convolution.
        
        Parameters
        ----------
        x : torch.Tensor
            (batch, in_channels, *spatial_dims)
        output_shape : optional
            Desired output shape after inverse transform
        grid : list of torch.Tensors, optional
            List of coordinate arrays for each dimension. If provided, must have length = self.order.
            If None, assumes unit spacing.

        Returns
        -------
        torch.Tensor
            Convolved output
        """
        batchsize, channels, *mode_sizes = x.shape
        fft_dims = list(range(-self.order, 0))

        if self.xno_block_precision == "half":
            x = x.half()

        # Forward transform
        # For Laplace transform method, we do an FFT in all dimensions.
        # complex_data is allowed. If complex_data=False, we still do full fftn (not rfft)
        x_fft = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)

        if self.xno_block_precision == "mixed":
            x_fft = x_fft.chalf()

        if self.xno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat

        # Determine frequencies (lambda_i)
        # If no grid is given, assume unit spacing
        if grid is None:
            diffs = [1.0]*self.order
            coords = [torch.fft.fftfreq(mode_sizes[i], d=diffs[i], device=x.device)*2*np.pi*1j for i in range(self.order)]
        else:
            # grid[i] shape: must be something like 1D array for each dimension
            # We'll assume grid[i] is 1D and evenly spaced.
            diffs = []
            coords = []
            for i in range(self.order):
                g = grid[i].to(x.device)
                d = (g[1]-g[0]).item()
                diffs.append(d)
                omega = torch.fft.fftfreq(mode_sizes[i], d=d, device=x.device)*2*np.pi*1j
                coords.append(omega)

        # truncate modes if needed
        # similar logic as in Fourier class:
        truncated_sizes = [min(s, n) for s, n in zip(mode_sizes, self.n_modes)]

        # Extract poles and residues
        weights_residue = self.weights_residue.to_tensor() if not torch.is_tensor(self.weights_residue) else self.weights_residue
        poles = self.get_poles()  # list of pole tensors, each (in_channels, out_channels, n_modes[i]) or (in_channels, n_modes[i]) if separable

        # We will slice x_fft and weights_residue similarly to how we do in the Fourier class:
        starts = [(max_modes - tsize) for (tsize, max_modes) in zip(truncated_sizes, self.max_n_modes)]
        # slices for residue and x_fft
        if self.separable:
            slices_w = [slice(None)] # (channels)
        else:
            slices_w = [slice(None), slice(None)] # (in_channels, out_channels)
        # handle last dimension differently if not complex_data?
        # Actually, for Laplace we generally consider a full FFT (complex_data mode). 
        # We'll assume full complex data scenario for simplicity.
        slices_w += [slice(st//2 if st%2==0 else (st//2), 
                           None if st==0 else (-st//2)) for st in starts]

        weight_res_sub = weights_residue[tuple(slices_w)]
        alpha_sub = x_fft[(slice(None), slice(None)) + tuple(
            slice(st//2 if st%2==0 else (st//2), None if st==0 else (-st//2)) for st in starts)]

        # Create meshgrid of lambda_i
        truncated_shape = alpha_sub.shape[2:]  # shape of truncated freq domain
        lambda_coords = []
        for i, c in enumerate(coords):
            # extract exactly truncated_shape[i] frequencies
            start_freq = starts[i]//2 if starts[i]%2==0 else (starts[i]//2)
            end_freq = None if starts[i] == 0 else (-starts[i]//2)
            lambda_coords.append(c[start_freq:end_freq])

        lambda_grid = torch.meshgrid(*lambda_coords, indexing='ij')  # each lambda_grid[i] has shape truncated_shape

        # Expand residues and compute Hw:
        Hw = weight_res_sub
        # Broadcast and apply pole-residue formula: Hw *= ∏_i 1/(lambda_grid[i]-weights_pole[i])
        for i in range(self.order):
            w_pole_i = poles[i]
            if isinstance(w_pole_i, FactorizedTensor):
                w_pole_i = w_pole_i.to_tensor()

            # Now we must broadcast w_pole_i to match Hw:
            # Hw shape: (in_channels, out_channels, *truncated_shape) or (channels, *truncated_shape) if separable
            # w_pole_i shape: (in_channels, out_channels, n_modes[i]) or (channels, n_modes[i])
            # Insert dimensions of size 1 around w_pole_i to match all dims:
            # We place the n_modes[i] dimension in the correct position:
            shape_expander = [1]*(2+self.order if not self.separable else 1+self.order)
            shape_expander[1 if self.separable else (2+i)] = w_pole_i.shape[-1]

            w_pole_i_expanded = w_pole_i.reshape(*w_pole_i.shape[:2] if not self.separable else w_pole_i.shape[:1],
                                                 *([1]*i), w_pole_i.shape[-1], *([1]*(self.order-i-1))) if not self.separable else \
                                w_pole_i.reshape(w_pole_i.shape[0], *([1]*i), w_pole_i.shape[-1], *([1]*(self.order-i-1)))

            # Now, lambda_grid[i] shape = truncated_shape, need to unsqueeze for batch and channels:
            # Hw: (B, In, ...), Actually Hw: (In, Out, ...) or (Ch,...)
            # Insert at front two dims for in/out if not separable:
            lambda_expanded = lambda_grid[i].unsqueeze(0).unsqueeze(0) if not self.separable else lambda_grid[i].unsqueeze(0)

            # Compute denominator
            denom = (lambda_expanded - w_pole_i_expanded)
            Hw = Hw / denom

        # Now we have Hw. 
        # Compute output_residues:
        # In the reference PR code, output_residue1 = sum over in_channels and frequencies of alpha*Hw
        # output_residue2 = similar but with sign depending on PDE vs ODE. We'll assume PDE: output_residue2 = -Hw * alpha.
        # We can just contract alpha_sub and Hw similarly as Fourier: alpha_sub (B,In,...) and Hw (In,Out,...)
        # For separable: (B,Ch,...) and (Ch,...) -> (B,Ch,...)
        # For non-separable: (B,In,...) and (In,Out,...) -> (B,Out,...)

        output_residue1 = _contract_dense(alpha_sub, Hw, separable=self.separable)
        # Let's define Pk = -Hw for PDE scenario
        Pk = -Hw
        output_residue2 = _contract_dense(alpha_sub, Pk, separable=self.separable)

        # Inverse FFT for transient response:
        x1 = torch.fft.ifftn(output_residue1, s=tuple(mode_sizes), dim=fft_dims, norm=self.fft_norm)
        x1 = torch.real(x1)

        # Steady-state response:
        # In provided code, steady-state involves exponentials: exp(weights_pole[i]*grid[i]).
        # For simplicity, assume the same shapes and that grid arrays for inverse transform are given.
        # If no grid provided, assume an array of shape mode_sizes[i].
        # We must compute:
        # x2 = real( einsum(output_residue2, exp( sum over i weights_pole[i]*grid[i] )) )
        # This is complex: we must form the exponential terms dimension-wise.
        # We'll form a combined exponential factor:
        # exp_factor = product over i of exp(weights_pole[i] * grid[i_dim])
        # grid for spatial domain: if grid not provided, create a simple range(0, mode_sizes[i]) as spatial domain.
        if grid is None:
            spatial_grids = [torch.arange(s, device=x.device, dtype=torch.float32) for s in mode_sizes]
        else:
            # Assuming grid[i] are the spatial/time coordinates
            spatial_grids = grid

        # Create a full meshgrid of spatial coordinates:
        space_mesh = torch.meshgrid(*spatial_grids, indexing='ij')  # each has shape mode_sizes
        # Construct the exponential term:
        # For each i, we do exp(weights_pole[i]*space_coordinate)
        # weights_pole[i] shape: broadcast similarly as before to get (in_channels, out_channels, *mode_sizes_of_freq, *mode_sizes_of_space)
        # We must be careful with shapes: output_residue2 has shape (B,Out,...freqs...) and we must match with exponentials.
        # Actually, for a large PDE scenario, one might integrate differently. The provided code does a direct exponential integral.
        # We'll mimic the structure from the provided code:
        # In 1D:
        # x2 = real(einsum("bix,ioxz->boz", output_residue2, exp(...)))
        # For nD, we must do:
        # x2 = real(einsum over frequency modes with output_residue2 and exponentials from poles and space)
        # We'll form an nD exponential factor by multiplying all exp terms dimension-wise.

        # Construct exponential factors:
        exp_factor = torch.ones(self.out_channels if not self.separable else self.in_channels, *mode_sizes, device=x.device, dtype=torch.cfloat)
        for i in range(self.order):
            w_pole_i = poles[i]
            if isinstance(w_pole_i, FactorizedTensor):
                w_pole_i = w_pole_i.to_tensor()
            # expand w_pole_i over space and other freq dims:
            w_pole_i_expanded = w_pole_i.reshape(*w_pole_i.shape[:2] if not self.separable else w_pole_i.shape[:1],
                                                 *([1]*i), w_pole_i.shape[-1],
                                                 *([1]*(self.order-i-1))) if not self.separable else \
                                w_pole_i.reshape(w_pole_i.shape[0],
                                                 *([1]*i), w_pole_i.shape[-1],
                                                 *([1]*(self.order-i-1)))
            # expand again for space dimensions (same size as mode_sizes)
            # We'll need to broadcast w_pole_i_expanded to shape (out_channels, *freq_modes, *space_modes)
            # Just unsqueeze space dims:
            for _ in range(self.order):
                w_pole_i_expanded = w_pole_i_expanded.unsqueeze(self.order+ (0 if self.separable else 1))
            # now w_pole_i_expanded shape ~ (In,Out,...freq...,1... for space)
            # space_mesh[i] shape: mode_sizes[i]
            # broadcast with exp
            # We'll apply exp(w_pole_i * space_mesh[i]) dimension by dimension:
            # Need to rearrange so that w_pole_i_expanded matches space_mesh[i]
            # Just expand space_mesh[i] and multiply:
            space_exp = torch.exp(w_pole_i_expanded * space_mesh[i].to(torch.complex64).unsqueeze(0 if self.separable else 0).unsqueeze(0 if not self.separable else 0))
            # Accumulate:
            exp_factor = exp_factor * space_exp

        # Now contract output_residue2 with exp_factor over frequency dims:
        # output_residue2 shape: (B, out_channels, *freq_modes)
        # exp_factor shape: (out_channels, *freq_modes, *space_modes) or (channels, *freq_modes, *space_modes)
        # Result shape: (B, out_channels, *space_modes)
        # eq: "bofreq,...freqspace->bospace"
        # We'll just do a tl.einsum:
        freq_dims = ''.join(einsum_symbols[2:2+self.order])
        space_dims = ''.join(einsum_symbols[2+self.order:2+2*self.order])
        # For out_channels dimension:
        if self.separable:
            eq = f'b{einsum_symbols[1]}{freq_dims},{einsum_symbols[1]}{freq_dims}{space_dims}->b{einsum_symbols[1]}{space_dims}'
        else:
            eq = f'b{einsum_symbols[1]}{freq_dims},{einsum_symbols[1]}{einsum_symbols[2]}{freq_dims}{space_dims}->b{einsum_symbols[2]}{space_dims}'

        x2 = tl.einsum(eq, output_residue2, exp_factor)
        x2 = torch.real(x2)
        # Normalize as in the code: dividing by product of sizes
        normalization = np.prod(mode_sizes)
        x2 = x2 / normalization

        x_out = x1 + x2

        if self.bias is not None:
            x_out = x_out + self.bias

        return x_out