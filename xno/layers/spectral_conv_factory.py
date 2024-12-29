# MIT License
# Copyright (c) 2024 Saman Pordanesh

import abc
import inspect
from typing import Dict, List, Union
from . import get_spectral_convolution_class
import torch.nn.functional as F
import torch
from .complex import CGELU

class BaseConvFactory(abc.ABC):
    """
    Abstract base class for all spectral convolution factories.
    Subclasses must implement:
        - select_conv_class()
        - get_extra_args()
        - Possibly override the update_norm() if that transformation changes the default norm.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels,
        n_modes: List[int], 
        transformation_kwargs: Dict, 
        norm: str=None, 
        complex_data: bool=False
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.dim = len(n_modes)
        self.transformation_kwargs = transformation_kwargs
        self.norm = norm  # Could be None, "group_norm", "instance_norm", etc.
        self.complex_data = complex_data

    @abc.abstractmethod
    def select_conv_class(self):
        """Return the class for the relevant spectral convolution (e.g. SpectralConvFourier)."""
        pass

    @abc.abstractmethod
    def get_extra_args(self) -> Dict:
        """
        Return a dictionary of extra arguments to pass to the conv constructor.
        Subclasses implement transformation-specific logic here.
        """
        pass

    def update_norm(self) -> str:
        """
        Some transformations (like LNO) might require you to use 'group_norm' if not specified.
        Subclasses can override if needed.
        Otherwise, by default, do nothing.
        """
        return self.norm
    
    def validate(self):
        """Base validation logic (can be overridden in subclasses)."""
        pass
    
    def non_linearity(self):
        """Return convolution specific non-linearity."""
        pass


class FNOConvFactory(BaseConvFactory):
    """Factory for Fourier-based Convolutions (FNO)."""
    def select_conv_class(self):
        # Return the relevant class
        return get_spectral_convolution_class(transformation="fno", dim=self.dim)

    def get_extra_args(self) -> Dict:
        # Usually no special wavelet/laplace args needed. We can just return {}
        return {}
    
    def validate(self):
        return super().validate()
    
    def non_linearity(self):
        # apply real nonlin if data is real, otherwise CGELU
        if self.complex_data: return CGELU
        else: return F.gelu
    
class HNOConvFactory(BaseConvFactory):
    """Factory for Hilbert-based Convolutions (HNO)."""
    def select_conv_class(self):
        return get_spectral_convolution_class(transformation="hno", dim=self.dim)

    def get_extra_args(self) -> Dict:
        return {}
    
    def validate(self):
        return super().validate()
    
    def non_linearity(self):
        # apply real nonlin if data is real, otherwise CGELU
        if self.complex_data: return CGELU
        else: return F.gelu


class LNOConvFactory(BaseConvFactory):
    """Factory for Laplace-based Convolutions (LNO)."""
    def select_conv_class(self):
        if 1 <= self.dim <= 3:
            return get_spectral_convolution_class(transformation="lno", dim=self.dim)
        else:
            raise ValueError(f"LNO supports 1D, 2D, or 3D. Got {self.dim}D.")

    def get_extra_args(self) -> Dict:
        return {}

    def update_norm(self) -> str:
        # LNO might require 'group_norm' if norm is None
        if self.norm is None:
            return "group_norm"
        return self.norm
    
    def non_linearity(self):
        return torch.sin

class WNOConvFactory(BaseConvFactory):
    """Factory for Wavelet-based Convolutions (WNO)."""
    def select_conv_class(self):
        if 1 <= self.dim <= 3:
            return get_spectral_convolution_class(transformation="wno", dim=self.dim)
        else:
            raise ValueError(f"WNO supports 1D, 2D, or 3D. Got {self.dim}D.")

    def get_extra_args(self) -> Dict:
        # Check mandatory wavelet args
        if "wavelet_level" not in self.transformation_kwargs:
            raise ValueError("Missing 'wavelet_level' in transformation_kwargs for WNO.")
        if "wavelet_size" not in self.transformation_kwargs:
            raise ValueError("Missing 'wavelet_size' in transformation_kwargs for WNO.")
        if self.transformation_kwargs["wavelet_level"] <= 0:
            raise ValueError("'wavelet_level' must be a positive integer.")

        # Build extra args 
        wavelet_level = self.transformation_kwargs["wavelet_level"]
        wavelet_size = self.transformation_kwargs["wavelet_size"]
        wavelet_filter = self.transformation_kwargs.get("wavelet_filter", None)
        wavelet_mode = self.transformation_kwargs.get("wavelet_mode", None)

        # Filter out args that aren't in the conv module signature
        conv_class = self.select_conv_class()
        sig = inspect.signature(conv_class.__init__)
        supported_args = sig.parameters.keys()

        tmp_args = {
            "wavelet_level": wavelet_level,
            "wavelet_size": wavelet_size,
            "wavelet_filter": wavelet_filter,
            "wavelet_mode": wavelet_mode,
        }
        extra_args = {k: v for k, v in tmp_args.items() if k in supported_args and v is not None}
        return extra_args
    
    def validate(self):
        return super().validate()
    
    def non_linearity(self):
        return F.mish

class SpectralConvFactory:
    """
    A high-level factory that chooses which sub-factory to instantiate
    based on the transformation type (FNO, HNO, LNO, WNO, etc.).
    """
    def __init__(
        self,
        in_channels, 
        out_channels,
        transformation: str,
        n_modes: List[int],
        norm: str,
        transformation_kwargs: Dict,
        complex_data = False,
        verbose=True,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.transformation = transformation.lower()
        self.n_modes = n_modes
        self.norm = norm
        self.dim = len(self.n_modes)
        self.transformation_kwargs = transformation_kwargs or {}
        self.verbose = verbose
        self.complex_data = complex_data
        
        # Error handeling for complex input data. 
        if self.complex_data and transformation.lower() in {"lno", "hno", "wno"}:
            raise ValueError("HNO, WNO, and LNO just work with real input values, for now!")

    def create_factory(self) -> BaseConvFactory:
        """Return an instance of the correct sub-factory."""
        factory = None
        if self.transformation == "fno":
            factory = FNOConvFactory(
                self.in_channels, 
                self.out_channels, 
                self.n_modes, 
                self.transformation_kwargs, 
                self.norm, 
                self.complex_data
                )
        elif self.transformation == "hno":
            factory = HNOConvFactory(
                self.in_channels, 
                self.out_channels, 
                self.n_modes, 
                self.transformation_kwargs, 
                self.norm, 
                self.complex_data
                )
        elif self.transformation == "lno":
            factory = LNOConvFactory(
                self.in_channels, 
                self.out_channels, 
                self.n_modes, 
                self.transformation_kwargs, 
                self.norm
                )
        elif self.transformation == "wno":
            factory = WNOConvFactory(
                self.in_channels, 
                self.out_channels, 
                self.n_modes, 
                self.transformation_kwargs, 
                self.norm
                )
        else:
            raise ValueError(
                f"Unknown transform type '{self.transformation}'. "
                "Supported transformations: FNO, HNO, LNO, WNO."
            )
            
        if self.verbose:
            print("======== Selected Kernel Description =======")
            print(f"Dimentionality: {self.dim}D")
            print(self.describe())
            config = (
                "================== Config ==================\n"
                f">>> Normaliztion: {factory.update_norm() or None}\n"
                ">>> Activation Function: \n"
                "============================================\n"
            )
            print(config)
            
        return factory

    def describe(self) -> str:
        """
        Return a description of the selected spectral convolution kernel.
        """
        description = CONV_DESCRIPTIONS.get(self.transformation, "Unknown transformation type.")
        return description

# descriptions.py (new module or within spectral_conv_factory.py)
CONV_DESCRIPTIONS = {
    "fno": (
        "Transformation: [ Fourier Neural Operator (FNO) Kernel ]\n"
        ">>> Overview:\n"
        "The FNO leverages Fourier Transform to map input data into the spectral domain, where\n"
        "convolutional operations are performed by truncating high-frequency modes.\n\n"
        ">>> Key Features:\n"
        "- Effective for parameterized Partial Differential Equations (PDEs).\n"
        "- Reduces computational complexity by retaining only significant modes.\n\n"
        ">>> Reference:\n"
        "Li, Z. et al. 'Fourier Neural Operator for Parametric Partial Differential Equations' (ICLR 2021).\n"
        "Link: https://arxiv.org/pdf/2010.08895\n"
        "============================================\n"
    ),
    "hno": (
        "Transformation: [ Hilbert Neural Operator (HNO) Kernel ]\n"
        ">>> Overview:\n"
        "The HNO applies Hilbert Transform, emphasizing the phase-shifted features of the input\n"
        "signal for enhanced data representation.\n\n"
        ">>> Key Features:\n"
        "- Focuses on phase information, useful in signal processing.\n"
        "- Suitable for scenarios requiring advanced spectral analysis.\n\n"
        ">>> Reference:\n"
        "This is an experimental implementation. Currently no formal reference.\n"
        "============================================\n"
    ),
    "lno": (
        "Transformation: [ Laplace Neural Operator (LNO) Kernel ]\n"
        ">>> Overview:\n"
        "The LNO uses a pole-residue formulation to compute solutions to PDEs in the Laplace domain.\n"
        "This kernel is highly effective for problems requiring stability and steady-state solutions.\n\n"
        ">>> Key Features:\n"
        "- Specially designed for systems dominated by Laplacian dynamics.\n"
        "- Balances transient and steady-state components.\n\n"
        ">>> Reference:\n"
        "Cao, Q. et al. 'LNO: Laplace Neural Operator for Solving Differential Equations'.\n"
        "Link: https://arxiv.org/pdf/2303.10528\n"
        "============================================\n"
    ),
    "wno": (
        "Transformation: [ Wavelet Neural Operator (WNO) Kernel ]\n"
        ">>> Overview:\n"
        "The WNO uses wavelet transformations to extract multi-resolution features from input signals.\n"
        "Wavelet decomposition offers a unique advantage in capturing localized features in both spatial\n"
        "and frequency domains.\n\n"
        ">>> Key Features:\n"
        "- Multi-resolution analysis via wavelet decomposition.\n"
        "- Supports both compressive sensing and hierarchical learning.\n\n"
        ">>> Reference:\n"
        "Tripura, T. et al. 'Wavelet neural operator: a neural operator for parametric partial differential equations'.\n"
        "Link: https://arxiv.org/pdf/2205.02191\n"
        "============================================\n"
    ),
}