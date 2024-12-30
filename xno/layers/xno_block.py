# MIT License
# Copyright (c) 2024 Saman Pordanesh
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software...

from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .complex import CGELU, apply_complex, ctanh, ComplexValued
from .normalization_layers import AdaIN, InstanceNorm
from .skip_connections import skip_connection

# from .spectral_convolution_x import SpectralConv
# from .spectral_convolution_fourier import SpectralConvFourier
# from .spectral_convolution_hilbert import SpectralConvHilbert
# from .spectral_convolution_laplace import SpectralConvLaplace1D, SpectralConvLaplace2D, SpectralConvLaplace3D
# from .spectral_convolution_wavelet import SpectralConvWavelet1D, SpectralConvWavelet2D, SpectralConvWavelet2DCwt, SpectralConvWavelet3D

from .spectral_conv_factory import SpectralConvFactory
from ..utils import validate_scaling_factor
import inspect


Number = Union[int, float]


class XNOBlocks(nn.Module):
    """XNOBlocks
    Parameters
    ----------
    in_channels : int
        input channels to Fourier layers
    out_channels : int
        output channels after Fourier layers
    n_modes : int, List[int]
        number of modes to keep along each dimension 
        in frequency space. Can either be specified as
        an int (for all dimensions) or an iterable with one
        number per dimension
    transformation : str
        The mathematical transformation ("X..") option as the convolution kernel. 
        This is what X stands on XNO. 
    transformation_kwargs : dict
        For extra keyword prompting where the spectral convolution kernel needs more tailored configuration. 
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        factor by which to scale outputs for super-resolution, by default None
    n_layers : int, optional
        number of Fourier layers to apply in sequence, by default 1
    max_n_modes : int, List[int], optional
        maximum number of modes to keep along each dimension, by default None
    xno_block_precision : str, optional
        floating point precision to use for computations, by default "full"
    channel_mlp_dropout : int, optional
        dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        nonlinear activation function to use between layers, by default None -> Default specified for different transformations at BaseConvFactory
    stabilizer : Literal["tanh"], optional
        stabilizing module to use between certain layers, by default None
        if "tanh", use tanh
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use, by default None
    ada_in_features : int, optional
        number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        whether to call forward pass with pre-activation, by default False
        if True, call nonlinear activation and norm before Fourier convolution
        if False, call activation and norms after Fourier convolutions
    xno_skip : str, optional
        module to use for XNO skip connections, by default "linear"
        see layers.skip_connections for more details
    channel_mlp_skip : str, optional
        module to use for ChannelMLP skip connections, by default "soft-gating"
        see layers.skip_connections for more details

    Other Parameters
    -------------------
    complex_data : bool, optional
        whether the XNO's data takes on complex values in space, by default False
    separable : bool, optional
        separable parameter for SpectralConv, by default False
    factorization : str, optional
        factorization parameter for SpectralConv, by default None
    rank : float, optional
        rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        module to use for convolutions in XNO block, by default SpectralConv
    joint_factorization : bool, optional
        whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        implementation parameter for SpectralConv, by default "factorized"
    decomposition_kwargs : _type_, optional
        kwargs for tensor decomposition in SpectralConv, by default dict()

    References
    -----------
    
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        transformation="FNO",
        transformation_kwargs=None,
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        xno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=None,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        xno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=None,
        fixed_rank_modes=False, #undoc
        implementation="factorized", #undoc
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)
        
        # self.hidden_channels = hidden_channels

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)
        
        self.transformation = transformation
        self.transformation_kwargs = transformation_kwargs

        self.max_n_modes = max_n_modes
        self.xno_block_precision = xno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.xno_skip = xno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data

        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features
        self.non_linearity = non_linearity
               
        extra_args = None  
        dim  = len(n_modes) 
                     
        # if self.transformation.lower() == "fno":
        #     conv_module = SpectralConvFourier
        # elif self.transformation.lower() == "hno":
        #     conv_module = SpectralConvHilbert
        # elif self.transformation.lower() == "lno":
        #     # Adding Laplace kernel special normaliazer. 
        #     if norm is None:
        #         norm = "group_norm"
        #     if dim == 1:
        #         conv_module = SpectralConvLaplace1D
        #     elif dim == 2:
        #         conv_module = SpectralConvLaplace2D
        #     elif dim == 3:
        #         conv_module = SpectralConvLaplace3D
        #     else: 
        #         raise ValueError(f"Dimensions must be 1D, 2D or 3D. You've passed n_modes for {dim} dimensions.")
            
        # elif self.transformation.lower() == "wno":
            
        #     if dim == 1:
        #         conv_module = SpectralConvWavelet1D
        #     elif dim == 2:
        #         conv_module = SpectralConvWavelet2D
        #     elif dim == 3:
        #         conv_module = SpectralConvWavelet3D
        #     else: 
        #         raise ValueError(f"Dimensions must be 1D, 2D or 3D. You've passed n_modes for {dim} dimensions.")
            
            
        #     # Dynamically filter arguments for the conv_module
        #     conv_signature = inspect.signature(conv_module.__init__)
        #     conv_supported_args = conv_signature.parameters.keys()

        #     # Check if transformation_kwargs is provided
        #     if self.transformation_kwargs is None:
        #         raise ValueError(
        #             "Missing `transformation_kwargs` for WNO. "
        #             "Expected a dictionary with keys: 'wavelet_level', 'wavelet_size', 'wavelet_filter', 'wavelet_mode'."
        #         ) 
        #     if "wavelet_level" not in self.transformation_kwargs:
        #         raise ValueError("Missing mandatory argument `wavelet_level` in `transformation_kwargs` for WNO.")
        #     if not isinstance(self.transformation_kwargs["wavelet_level"], int) or self.transformation_kwargs["wavelet_level"] <= 0:
        #         raise ValueError("`wavelet_level` must be a positive integer.")
            
        #     if "wavelet_size" not in self.transformation_kwargs:
        #         raise ValueError("Missing mandatory argument `wavelet_size` in `transformation_kwargs` for WNO.")
        #     if not isinstance(self.transformation_kwargs["wavelet_size"], list) or len(self.transformation_kwargs["wavelet_size"]) != 2:
        #         raise ValueError("`wavelet_size` must be a list of two integers (e.g., [32, 32]).")
            
        #     # Extract or use defaults for optional arguments
        #     wavelet_filter = self.transformation_kwargs.get("wavelet_filter", None)
        #     wavelet_mode = self.transformation_kwargs.get("wavelet_mode", None)

        #     # Prepare arguments for the constructor
        #     extra_args = {
        #         "wavelet_level": self.transformation_kwargs["wavelet_level"],
        #         "wavelet_size": self.transformation_kwargs["wavelet_size"],
        #     }
            
        #     # Add optional arguments only if they are explicitly provided
        #     if wavelet_filter is not None:
        #         extra_args["wavelet_filter"] = wavelet_filter
        #     if wavelet_mode is not None:
        #         extra_args["wavelet_mode"] = wavelet_mode
                
        #     extra_args = {k: v for k, v in extra_args.items() if k in conv_supported_args and v is not None}

        # else:
        #     raise ValueError(
        #         f"Unknown transform type '{transformation}'. "
        #         "XNO just accepts FNO, HNO, LNO, and WNO as transformation argument."
        #     )
        
        
         # Decide if we are auto-selecting the spectral conv or using a user-supplied one
        if conv_module is None:
            # Use the new OOP factory
            factory = SpectralConvFactory(
                in_channels = in_channels, 
                out_channels = out_channels,
                transformation=self.transformation,
                n_modes=self._n_modes,
                norm=norm,
                transformation_kwargs=self.transformation_kwargs,
                complex_data = self.complex_data
            )
            sub_factory = factory.create_factory()  # e.g. WNOConvFactory, LNOConvFactory, ...
            sub_factory.validate()
            conv_module = sub_factory.select_conv_class()
            extra_args = sub_factory.get_extra_args()
            # Possibly update 'norm' if needed
            norm = sub_factory.update_norm()
            # Retrieve transformation specifc non-linearity 
            if self.non_linearity is None:
                self.non_linearity = sub_factory.non_linearity()
        else:
            # user manually gave a conv, so no special logic
            conv_module = conv_module
            extra_args = {}


        # apply real nonlin if data is real, otherwise CGELU
        # if self.complex_data:
        #     self.non_linearity = CGELU
        # else:
        #     self.non_linearity = non_linearity
                    
        self.convs = nn.ModuleList([
                conv_module(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                n_modes=self.n_modes,
                resolution_scaling_factor=None if resolution_scaling_factor is None else self.resolution_scaling_factor[i],
                max_n_modes=max_n_modes,
                rank=rank,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                xno_block_precision=xno_block_precision,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=complex_data,
                **extra_args
            ) 
            for i in range(n_layers)])

        self.xno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=xno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.xno_skips = nn.ModuleList(
                [ComplexValued(x) for x in self.xno_skips]
                )

        self.channel_mlp = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(self.out_channels * channel_mlp_expansion),
                    dropout=channel_mlp_dropout,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.channel_mlp = nn.ModuleList(
                [ComplexValued(x) for x in self.channel_mlp]
            )

        self.channel_mlp_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=channel_mlp_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        if self.complex_data:
            self.channel_mlp_skips = nn.ModuleList(
                [ComplexValued(x) for x in self.channel_mlp_skips]
            )

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                    [
                        InstanceNorm()
                        for _ in range(n_layers * self.n_norms)
                    ]
                )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, ada_in]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_xno = self.xno_skips[index](x)
        
        x_skip_xno = self.convs[index].transform(x_skip_xno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_xno = self.convs[index](x, output_shape=output_shape)
        #self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_xno = self.norm[self.n_norms * index](x_xno)

        x = x_xno + x_skip_xno

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_xno = self.xno_skips[index](x)
        x_skip_xno = self.convs[index].transform(x_skip_xno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_xno = self.convs[index](x, output_shape=output_shape)

        x = x_xno + x_skip_xno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-XNO Block layer from the jointly parametrized main block

        The parametrization of an XNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)
    
"""
Statndard N-Dim pytorch normalizer. 
This helps avoid exponential growth after each convolution in SpectralConv class (if needed). 
"""
class NDNormalizer(nn.Module):
    def __init__(self, num_channels):
        super(NDNormalizer, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_channels)

    def forward(self, x):
        return self.norm(x)