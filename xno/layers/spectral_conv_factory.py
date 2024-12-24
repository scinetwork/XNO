# MIT License
# Copyright (c) 2024 Saman Pordanesh
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software...

import abc
import inspect
from typing import Dict, List, Union
from . import get_spectral_convolution_class

class BaseConvFactory(abc.ABC):
    """
    Abstract base class for all spectral convolution factories.
    Subclasses must implement:
        - select_conv_class()
        - get_extra_args()
        - Possibly override the update_norm_if_needed() if that transformation changes the default norm.
    """

    def __init__(
        self, 
        n_modes: List[int], 
        transformation_kwargs: Dict, 
        norm: str=None
    ):
        self.n_modes = n_modes
        self.dim = len(n_modes)
        self.transformation_kwargs = transformation_kwargs
        self.norm = norm  # Could be None, "group_norm", "instance_norm", etc.

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

    def update_norm_if_needed(self) -> str:
        """
        Some transformations (like LNO) might require you to use 'group_norm' if not specified.
        Subclasses can override if needed.
        Otherwise, by default, do nothing.
        """
        return self.norm


class FNOConvFactory(BaseConvFactory):
    """Factory for Fourier-based Convolutions (FNO)."""
    def select_conv_class(self):
        # Return the relevant class
        return get_spectral_convolution_class(transformation="fno", dim=self.dim)

    def get_extra_args(self) -> Dict:
        # Usually no special wavelet/laplace args needed. We can just return {}
        return {}


class HNOConvFactory(BaseConvFactory):
    """Factory for Hilbert-based Convolutions (HNO)."""
    def select_conv_class(self):
        return get_spectral_convolution_class(transformation="hno", dim=self.dim)

    def get_extra_args(self) -> Dict:
        return {}


class LNOConvFactory(BaseConvFactory):
    """Factory for Laplace-based Convolutions (LNO)."""
    def select_conv_class(self):
        if 1 <= self.dim <= 3:
            return get_spectral_convolution_class(transformation="lno", dim=self.dim)
        else:
            raise ValueError(f"LNO supports 1D, 2D, or 3D. Got {self.dim}D.")

    def get_extra_args(self) -> Dict:
        return {}

    def update_norm_if_needed(self) -> str:
        # LNO might require 'group_norm' if norm is None
        if self.norm is None:
            return "group_norm"
        return self.norm


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


class SpectralConvFactory:
    """
    A high-level factory that chooses which sub-factory to instantiate
    based on the transformation type (FNO, HNO, LNO, WNO, etc.).
    """
    def __init__(
        self,
        transformation: str,
        n_modes: List[int],
        norm: str,
        transformation_kwargs: Dict
    ):
        self.transformation = transformation.lower()
        self.n_modes = n_modes
        self.norm = norm
        self.transformation_kwargs = transformation_kwargs or {}

    def create_factory(self) -> BaseConvFactory:
        """Return an instance of the correct sub-factory."""
        if self.transformation == "fno":
            return FNOConvFactory(self.n_modes, self.transformation_kwargs, self.norm)
        elif self.transformation == "hno":
            return HNOConvFactory(self.n_modes, self.transformation_kwargs, self.norm)
        elif self.transformation == "lno":
            return LNOConvFactory(self.n_modes, self.transformation_kwargs, self.norm)
        elif self.transformation == "wno":
            return WNOConvFactory(self.n_modes, self.transformation_kwargs, self.norm)
        else:
            raise ValueError(
                f"Unknown transform type '{self.transformation}'. "
                "Supported transformations: FNO, HNO, LNO, WNO."
            )
