# MIT License
# Copyright (c) 2024 Saman Pordanesh
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software...

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple, Union
from .resample import resample
from ..utils import validate_scaling_factor


Number = Union[int, float]

try:
    import ptwt, pywt
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT 
except ImportError:
    raise ImportError("Required packages are missing. Please install dependencies as per the documentation.")



def _check_wavelet_mode_valid(mode: str):
    """
    Validate that wavelet mode is recognized by pywt.
    Raises a ValueError if not valid.
    """
    valid_modes = pywt.Modes.modes
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid wavelet mode '{mode}'. "
            f"Expected one of: {list(valid_modes.keys())}"
        )

def _check_wavelet_filter_valid(wavelet_filter: str):
    """
    Validate that the wavelet_filter is recognized by pywt.
    Raises a ValueError if not valid.
    """
    try:
        _ = pywt.Wavelet(wavelet_filter)  # Just to see if it raises
    except ValueError:
        raise ValueError(
            f"Wavelet filter '{wavelet_filter}' not recognized by pywt. "
            "Check valid wavelet names in pywt.wavelist()"
        )

def _check_wavelet_size_valid(
    wavelet_size: List[int], 
    wavelet_level: int, 
    wavelet_filter: str, 
    wavelet_mode: str
):
    """
    Validate wavelet_size and wavelet_level together.
    For a wavelet decomposition at level L, one typical rule-of-thumb is:
         size >= 2^L * (filter_length - 1)
    for each dimension.
    If this is not met, we raise an Exception.

    Note: This check can be stricter or more lenient depending on the wavelet mode.
          We enforce it strictly for safety.
    """
    w = pywt.Wavelet(wavelet_filter)
    # The "dec_len" is the length of the decomposition filter.
    filter_length = w.dec_len

    # For certain wavelets (biorthogonal, etc.), dec_len can be large.
    # Adjust logic if you have a custom approach.
    for dim_idx, dim_size in enumerate(wavelet_size):
        min_required = (2 ** wavelet_level) * (filter_length - 1)
        if dim_size < min_required:
            raise ValueError(
                f"Dimension {dim_idx} of wavelet_size = {dim_size} is too small for "
                f"wavelet_level = {wavelet_level} and wavelet_filter = '{wavelet_filter}' "
                f"(dec_len={filter_length}). Minimum required size for that dimension is "
                f"{min_required}, but got {dim_size}."
            )



""" Def: 1d Wavelet convolutional layer """
class SpectralConvWavelet1D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level: int, 
        wavelet_size: List[int], 
        wavelet_filter: List[str]=['db4'],
        wavelet_mode: str='symmetric',
        n_modes=None,
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
        device=None,    
        **kwargs,
    ):
        super(SpectralConvWavelet1D, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        wavelet_level        : scalar, levels of wavelet decomposition
        wavelet_size         : list[int], length of input 1D signal
        wavelet_filter      : list[string], wavelet filter
        wavelet_mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_level = wavelet_level
        if isinstance(wavelet_size, list):
            if len(wavelet_size) != 1:
                raise Exception('wavelet_size: WaveConv1d accepts the wavelet_size of 1D signal in list with 1 element')
            else:
                self.wavelet_size = wavelet_size[-1]
        else:
            raise Exception('wavelet_size: WaveConv1d accepts wavelet_size of 1D signal is list') 
        
        if wavelet_level < 1: raise ValueError(f"wavelet_level (J) must be >= 1, got {wavelet_level}")
        
        if not wavelet_filter or not isinstance(wavelet_filter, list):
            raise ValueError(
                "wavelet_filter must be a non-empty list of wavelet names. "
                f"Got: {wavelet_filter}"
            )

        self.wavelet_filter = wavelet_filter[0]
        self.wavelet_mode = wavelet_mode
        
        # Check filter and mode validity 
        _check_wavelet_filter_valid(self.wavelet_filter)
        _check_wavelet_mode_valid(self.wavelet_mode)
        
        # This ensures that each dimension is big enough to do wavelet_level decompositions
        # _check_wavelet_size_valid(self.wavelet_size, self.wavelet_level, self.wavelet_filter, self.wavelet_mode)
        
        self.dwt_ = DWT1D(
            wave=self.wavelet_filter, 
            J=self.wavelet_level, 
            mode=self.wavelet_mode
        )
        dummy_data = torch.randn( 1,1,self.wavelet_size ) 
        mode_data, _ = self.dwt_(dummy_data)
        self.modes1 = mode_data.shape[-1]
        
        self.n_modes = (self.modes1)
        self.max_n_modes = self.n_modes
        
        self.order = self.n_modes
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)

        # Initializing the class wise global weights tensor
        self.scale = (1 / (in_channels*out_channels))
        self.weight = nn.Parameter(
            self.scale * torch.randn(
                2, 
                in_channels,
                out_channels, 
                self.modes1, 
            )
        )
    
    def transform(
        self, 
        x, 
        output_shape=None
        ):
        
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    # Convolution
    def mul1d(
        self, 
        input,
        weights
    ):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x ) 
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(
        self, 
        x: torch.Tensor, 
        output_shape: Optional[Tuple[int]] = None
    ):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x]
        
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        
        batchsize = x.shape[0]
        
        if x.shape[-1] > self.wavelet_size:
            factor = int(np.log2(x.shape[-1] // self.wavelet_size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(
                wave=self.wavelet_filter, 
                J=self.wavelet_level+factor, 
                mode=self.wavelet_mode
            ).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.wavelet_size:
            factor = int(np.log2(self.wavelet_size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(
                wave=self.wavelet_filter, 
                J=self.wavelet_level-factor, 
                mode=self.wavelet_mode
            ).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(
                wave=self.wavelet_filter, 
                J=self.wavelet_level, 
                mode=self.wavelet_mode
            ).to(x.device)
            x_ft, x_coeff = dwt(x)
                        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros(
            batchsize,
            self.out_channels, 
            x_ft.shape[-1],
            device= x.device
        )
        out_coeff = [
            torch.zeros(
                batchsize, 
                self.out_channels,
                coeffs.shape[-1], 
                device= x.device
            ) for coeffs in x_coeff
        ]

        # Dynamic modes handeling for different input x shpaes
        L_FT = x_ft.shape[-1]
        L_COE = x_coeff[-1].shape[-1]
        
        modes1_ft = min(self.modes1, L_FT)
        modes1_coe = min(self.modes1, L_COE)
        
        # Multiply the final low pass wavelet coefficients
        out_ft[:,:, :modes1_ft] = self.mul1d(
            x_ft[:,:, :modes1_ft], 
            self.weight[0][:,:, :modes1_ft]
        )
        
        # Multiply the final high pass wavelet coefficients
        out_coeff[-1][:,:, :modes1_coe] = self.mul1d(
            x_coeff[-1][:,:, :modes1_coe].clone(), 
            self.weight[1][:,:, :modes1_coe]
        )
    
        # Reconstruct the signal
        idwt = IDWT1D(
            wave=self.wavelet_filter, 
            mode=self.wavelet_mode
        ).to(x.device)
        
        x = idwt((out_ft, out_coeff)) 
        return x


""" Def: 2d Wavelet convolutional layer (discrete) """
class SpectralConvWavelet2D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level: int, 
        wavelet_size: List[int], 
        wavelet_filter: List[str]=['db4'],
        wavelet_mode: str='symmetric',
        n_modes=None,
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
        device=None,
        **kwargs,
    ):
        super(SpectralConvWavelet2D, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        wavelet_level        : scalar, levels of wavelet decomposition
        wavelet_size         : List[int], length of input 1D signal
        wavelet_filter      : List[str], wavelet filters
        wavelet_mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Horizontal-Detailed wavelet coefficients
        self.weights3 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Vertical-Detailed wavelet coefficients
        self.weights4 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Diagonal-Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_level = wavelet_level
        if isinstance(wavelet_size, list):
            if len(wavelet_size) != 2:
                raise Exception('wavelet_size: WaveConv2d accepts the wavelet_size of 2D signal in list with 2 elements')
            else:
                self.wavelet_size = wavelet_size
        else:
            raise Exception('wavelet_size: WaveConv2d accepts wavelet_size of 2D signal is list')
        
        if wavelet_level < 1: raise ValueError(f"wavelet_level (J) must be >= 1, got {wavelet_level}")
        
        if not wavelet_filter or not isinstance(wavelet_filter, list):
            raise ValueError(
                "wavelet_filter must be a non-empty list of wavelet names. "
                f"Got: {wavelet_filter}"
            )
        
        self.wavelet_filter = wavelet_filter[0]       
        self.wavelet_mode = wavelet_mode
        
        # Check filter and mode validity 
        _check_wavelet_filter_valid(self.wavelet_filter)
        _check_wavelet_mode_valid(self.wavelet_mode)
        
        # This ensures that each dimension is big enough to do wavelet_level decompositions
        # _check_wavelet_size_valid(self.wavelet_size, self.wavelet_level, self.wavelet_filter, self.wavelet_mode)
        
        
        dummy_data = torch.randn( 1,1,*self.wavelet_size )        
        dwt_ = DWT(
            J=self.wavelet_level, 
            mode=self.wavelet_mode, 
            wave=self.wavelet_filter
        )
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        self.n_modes = (self.modes1, self.modes2)
        self.max_n_modes = self.n_modes
        
        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)
                
        # Initializing the class wise global weights tensor
        self.scale = (1 / (in_channels * out_channels))
        self.weight = nn.Parameter(
            self.scale * torch.randn(
                4, 
                in_channels,
                out_channels, 
                self.modes1, 
                self.modes2
            )
        )
        
    def transform(
        self, 
        x, 
        output_shape=None
    ):
        
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    # Convolution
    def mul2d(
        self, 
        input, 
        weights
    ):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(
        self, 
        x: torch.Tensor, 
        output_shape: Optional[Tuple[int]] = None
    ):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        batchsize = x.shape[0]
        
        if x.shape[-1] > self.wavelet_size[-1]:
            factor = int(np.log2(x.shape[-1] // self.wavelet_size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(
                J=self.wavelet_level+factor, 
                mode=self.wavelet_mode, 
                wave=self.wavelet_filter
            ).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.wavelet_size[-1]:
            factor = int(np.log2(self.wavelet_size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(
                J=self.wavelet_level-factor, 
                mode=self.wavelet_mode, 
                wave=self.wavelet_filter
            ).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(
                J=self.wavelet_level, 
                mode=self.wavelet_mode, 
                wave=self.wavelet_filter
            ).to(x.device)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros(
            batchsize, 
            self.out_channels, 
            x_ft.shape[-2], 
            x_ft.shape[-1], 
            device=x.device
        )        
        out_coeff = [
            torch.zeros(
                batchsize,
                self.out_channels,
                coeffs.shape[-3],
                coeffs.shape[-2],
                coeffs.shape[-1], 
                device= x.device
            ) for coeffs in x_coeff
        ]
                   
        # Dynamic modes handeling for different input x shpaes
        H_FT, W_FT = x_ft.shape[-2], x_ft.shape[-1]
        H_COE, W_COE = x_coeff[-1].shape[-2], x_coeff[-1].shape[-1]
        
        modes1_ft = min(self.modes1, H_FT)
        modes2_ft = min(self.modes2, W_FT)
        modes1_coe = min(self.modes1, H_COE)
        modes2_coe = min(self.modes2, W_COE)
        
        # Multiply the final approximate Wavelet modes
        out_ft[:,:, :modes1_ft, :modes2_ft]  = self.mul2d(
            x_ft[:,:, :modes1_ft, :modes2_ft], 
            self.weight[0][:,:, :modes1_ft, :modes2_ft]
        )
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0, :modes1_coe, :modes2_coe] = self.mul2d(
            x_coeff[-1][:,:,0, :modes1_coe, :modes2_coe].clone(), 
            self.weight[1][:,:, :modes1_coe, :modes2_coe]
        )
        
        out_coeff[-1][:,:,1, :modes1_coe, :modes2_coe] = self.mul2d(
            x_coeff[-1][:,:,1, :modes1_coe, :modes2_coe].clone(), 
            self.weight[2][:,:, :modes1_coe, :modes2_coe]
        )
        
        out_coeff[-1][:,:,2, :modes1_coe, :modes2_coe] = self.mul2d(
            x_coeff[-1][:,:,2, :modes1_coe, :modes2_coe].clone(), 
            self.weight[3][:,:, :modes1_coe, :modes2_coe]
        )
        
        # Return to physical space        
        idwt = IDWT(
            mode=self.wavelet_mode, 
            wave=self.wavelet_filter
        ).to(x.device)
        x = idwt((out_ft, out_coeff))
        return x

    
""" Def: 2d Wavelet convolutional layer (slim continuous) """
class SpectralConvWavelet2DCwt(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level: int, 
        wavelet_size: List[int],
        wavelet_filter: List[str]=['near_sym_b', 'qshift_b'],
        n_modes=None,
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
        device=None,    
        **kwargs,
    ):
        super(SpectralConvWavelet2DCwt, self).__init__()

        """
        !! It is computationally expensive than the discrete "WaveConv2d" !!
        2D Wavelet layer. It does SCWT (Slim continuous wavelet transform),
                                linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        wavelet_level        : scalar, levels of wavelet decomposition
        wavelet_size         : List[int], length of input 1D signal
        wavelet_filter[0]     : string, Specifies the first level biorthogonal wavelet filters
        wavelet_filter[1]     : string, Specifies the second level quarter shift filters
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights- 15r, 45r, 75r, 105r, 135r, 165r : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for REAL wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        self.weights- 15c, 45c, 75c, 105c, 135c, 165c : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for COMPLEX wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_level = wavelet_level
        if isinstance(wavelet_size, list):
            if len(wavelet_size) != 2:
                raise Exception('wavelet_size: WaveConv2dCwt accepts the wavelet_size of 2D signal in list with 2 elements')
            else:
                self.wavelet_size = wavelet_size
        else:
            raise Exception('wavelet_size: WaveConv2dCwt accepts wavelet_size of 2D signal is list')
        
        if wavelet_level < 1: raise ValueError(f"wavelet_level (J) must be >= 1, got {wavelet_level}")
        
        if not wavelet_filter or not isinstance(wavelet_filter, list):
            raise ValueError(
                "wavelet_filter must be a non-empty list of wavelet names. "
                f"Got: {wavelet_filter}"
            )
        
        self.wavelet_filter1 = wavelet_filter[0]
        self.wavelet_filter2 = wavelet_filter[1] 
    
        
        dummy_data = torch.randn( 1,1,*self.wavelet_size ) 
        dwt_ = DTCWTForward(
            J=self.wavelet_level, 
            biort=self.wavelet_filter1, 
            qshift=self.wavelet_filter2
        )
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]
        
        self.n_modes = (self.modes1, self.modes2)
        self.max_n_modes = self.n_modes
        
        self.order = len(self.n_modes)   
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)
        
        # Initializing the class wise global weights tensor
        n_subbands = 13  # 1 approximate + 12 detail (6 angles × 2)
        self.scale = (1 / (in_channels * out_channels))
        self.weight = nn.Parameter(
            self.scale * torch.randn(
                n_subbands,
                in_channels,
                out_channels,
                max(self.modes1, self.modes21),  # height
                max(self.modes2, self.modes22)   # width
            )
        )
        
    def transform(
        self, 
        x, 
        output_shape=None
        ):
        
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    # Convolution
    def mul2d(
        self, 
        input, 
        weights
    ):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(
        self, 
        x: torch.Tensor, 
        output_shape: Optional[Tuple[int]] = None
    ):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """      
        batchsize = x.shape[0]
        
        if x.shape[-1] > self.wavelet_size[-1]:
            factor = int(np.log2(x.shape[-1] // self.wavelet_size[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(
                J=self.wavelet_level+factor, 
                biort=self.wavelet_filter1, 
                qshift=self.wavelet_filter2
            ).to(x.device)
            x_ft, x_coeff = cwt(x)
            
        elif x.shape[-1] < self.wavelet_size[-1]:
            factor = int(np.log2(self.wavelet_size[-1] // x.shape[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(
                J=self.wavelet_level-factor, 
                biort=self.wavelet_filter1, 
                qshift=self.wavelet_filter2
            ).to(x.device)
            x_ft, x_coeff = cwt(x)            
        else:
            # Compute dual tree continuous Wavelet coefficients 
            cwt = DTCWTForward(
                J=self.wavelet_level, 
                biort=self.wavelet_filter1, 
                qshift=self.wavelet_filter2
            ).to(x.device)
            x_ft, x_coeff = cwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros(
            batchsize, 
            self.out_channels, 
            x_ft.shape[-2], 
            x_ft.shape[-1], 
            device=x.device
        )
        out_coeff = [
            torch.zeros(
                batchsize,
                self.out_channels,
                coeffs.shape[-4],
                coeffs.shape[-3],
                coeffs.shape[-2],
                coeffs.shape[-1], 
                device= x.device
            ) for coeffs in x_coeff
        ]
                
        H_FT, W_FT = x_ft.shape[-2], x_ft.shape[-1]
        H_COE, W_COE = x_coeff[-1].shape[-2], x_coeff[-1].shape[-1]
        
        modes1_ft = min(self.modes1, H_FT)
        modes2_ft = min(self.modes2, W_FT)
        modes21_coe = min(self.modes21, H_COE)
        modes22_coe = min(self.modes21, W_COE)
        
        # Multiply the final approximate Wavelet modes        
        out_ft[..., :modes1_ft, :modes2_ft] = self.mul2d(
            x_ft[..., :modes1_ft, :modes2_ft],
            self.weight[0, :, :, :modes1_ft, :modes2_ft]
        )
                
        # Detail subbands (indices 1 to 12 in self.weight)
        out_coeff[-1][:,:,0, :modes21_coe, :modes22_coe, 0] = self.mul2d(
            x_coeff[-1][:,:,0, :modes21_coe, :modes22_coe, 0].clone(),
            self.weight[1, :, :, :modes21_coe, :modes22_coe]  # 15r
        )
        out_coeff[-1][:,:,0, :modes21_coe, :modes22_coe, 1] = self.mul2d(
            x_coeff[-1][:,:,0, :modes21_coe, :modes22_coe, 1].clone(),
            self.weight[2, :, :, :modes21_coe, :modes22_coe]  # 15c
        )

        out_coeff[-1][:,:,1, :modes21_coe, :modes22_coe, 0] = self.mul2d(
            x_coeff[-1][:,:,1, :modes21_coe, :modes22_coe, 0].clone(),
            self.weight[3, :, :, :modes21_coe, :modes22_coe]  # 45r
        )
        out_coeff[-1][:,:,1, :modes21_coe, :modes22_coe, 1] = self.mul2d(
            x_coeff[-1][:,:,1, :modes21_coe, :modes22_coe, 1].clone(),
            self.weight[4, :, :, :modes21_coe, :modes22_coe]  # 45c
        )

        out_coeff[-1][:,:,2, :modes21_coe, :modes22_coe, 0] = self.mul2d(
            x_coeff[-1][:,:,2, :modes21_coe, :modes22_coe, 0].clone(),
            self.weight[5, :, :, :modes21_coe, :modes22_coe]  # 75r
        )
        out_coeff[-1][:,:,2, :modes21_coe, :modes22_coe, 1] = self.mul2d(
            x_coeff[-1][:,:,2, :modes21_coe, :modes22_coe, 1].clone(),
            self.weight[6, :, :, :modes21_coe, :modes22_coe]  # 75c
        )

        out_coeff[-1][:,:,3, :modes21_coe, :modes22_coe, 0] = self.mul2d(
            x_coeff[-1][:,:,3, :modes21_coe, :modes22_coe, 0].clone(),
            self.weight[7, :, :, :modes21_coe, :modes22_coe]  # 105r
        )
        out_coeff[-1][:,:,3, :modes21_coe, :modes22_coe, 1] = self.mul2d(
            x_coeff[-1][:,:,3, :modes21_coe, :modes22_coe, 1].clone(),
            self.weight[8, :, :, :modes21_coe, :modes22_coe]  # 105c
        )

        out_coeff[-1][:,:,4, :modes21_coe, :modes22_coe, 0] = self.mul2d(
            x_coeff[-1][:,:,4, :modes21_coe, :modes22_coe, 0].clone(),
            self.weight[9, :, :, :modes21_coe, :modes22_coe]  # 135r
        )
        out_coeff[-1][:,:,4, :modes21_coe, :modes22_coe, 1] = self.mul2d(
            x_coeff[-1][:,:,4, :modes21_coe, :modes22_coe, 1].clone(),
            self.weight[10, :, :, :modes21_coe, :modes22_coe]  # 135c
        )

        out_coeff[-1][:,:,5, :modes21_coe, :modes22_coe, 0] = self.mul2d(
            x_coeff[-1][:,:,5, :modes21_coe, :modes22_coe, 0].clone(),
            self.weight[11, :, :, :modes21_coe, :modes22_coe]  # 165r
        )
        out_coeff[-1][:,:,5, :modes21_coe, :modes22_coe, 1] = self.mul2d(
            x_coeff[-1][:,:,5, :modes21_coe, :modes22_coe, 1].clone(),
            self.weight[12, :, :, :modes21_coe, :modes22_coe]  # 165c
        )
            
        
        # Reconstruct the signal
        icwt = DTCWTInverse(biort=self.wavelet_filter1, qshift=self.wavelet_filter2).to(x.device)
        x = icwt((out_ft, out_coeff))
        return x
    
    
""" Def: 3d Wavelet convolutional layer """
class SpectralConvWavelet3D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level: int, 
        wavelet_size: List[int], 
        wavelet_filter: List[str]=['db4'], 
        wavelet_mode: str='periodic',
        n_modes=None,
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
        device=None, 
        **kwargs,   
    ):
        super(SpectralConvWavelet3D, self).__init__()

        """
        3D Wavelet layer. It does 3D DWT, linear transform, and Inverse dWT.    
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        wavelet_level        : scalar, levels of wavelet decomposition
        wavelet_size         : List[int], length of input 1D signal
        wavelet_filter      : List[str], Specifies the first level biorthogonal wavelet filters
        wavelet_mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Approximate wavelet coefficients
        self.weights_ : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Detailed wavelet coefficients 
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_level = wavelet_level
        if isinstance(wavelet_size, list):
            if len(wavelet_size) != 3:
                raise Exception('wavelet_size: WaveConv3d accepts the wavelet_size of 3D signal in list with 3 elements')
            else:
                self.wavelet_size = wavelet_size
        else:
            raise Exception('wavelet_size: WaveConv3d accepts wavelet_size of 3D signal is list')
        
        if wavelet_level < 1: raise ValueError(f"wavelet_level (J) must be >= 1, got {wavelet_level}")
        
        if not wavelet_filter or not isinstance(wavelet_filter, list):
            raise ValueError(
                "wavelet_filter must be a non-empty list of wavelet names. "
                f"Got: {wavelet_filter}"
            )
        
        self.wavelet_filter = wavelet_filter[0]
        self.wavelet_mode = wavelet_mode
        
        # Check filter and mode validity 
        _check_wavelet_filter_valid(self.wavelet_filter)
        _check_wavelet_mode_valid(self.wavelet_mode)
        
        # This ensures that each dimension is big enough to do wavelet_level decompositions
        # _check_wavelet_size_valid(self.wavelet_size, self.wavelet_level, self.wavelet_filter, self.wavelet_mode)

        
        dummy_data = torch.randn([*self.wavelet_size]).unsqueeze(0)
        mode_data = wavedec3(
            dummy_data, 
            pywt.Wavelet(self.wavelet_filter), 
            level=self.wavelet_level, 
            mode=self.wavelet_mode
        )
        self.modes1 = mode_data[0].shape[-3]
        self.modes2 = mode_data[0].shape[-2]
        self.modes3 = mode_data[0].shape[-1]
        self.n_modes = (self.modes1, self.modes2, self.modes3)
        
        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)
        
        self.max_n_modes = self.n_modes
        
        # Initializing the class wise global weights tensor
        self.scale = (1 / (in_channels * out_channels))
        self.weight = nn.Parameter(
            self.scale * torch.randn(
                8, 
                in_channels,
                out_channels, 
                self.modes1, 
                self.modes2, 
                self.modes3
            )
        )
    
    def transform(
        self, 
        x, 
        output_shape=None
        ):
        
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    # Convolution
    def mul3d(
        self, 
        input, 
        weights
    ):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(in_channel * x * y * z)
                  3D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y * z)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(out_channel * x * y * z)
        """
        return torch.einsum("ixyz,ioxyz->oxyz", input, weights)

    def forward(
        self, 
        x: torch.Tensor, 
        output_shape: Optional[Tuple[int]] = None
    ):
        batchsize = x.shape[0]
        xr = torch.zeros(
            batchsize,
            self.out_channels,
            x.shape[-3],
            x.shape[-2], 
            x.shape[-1], 
            device = x.device
        )
        
        for i in range(x.shape[0]):
            
            if x.shape[-1] > self.wavelet_size[-1]:
                factor = int(np.log2(x.shape[-1] // self.wavelet_size[-1]))
                
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(
                    x[i, ...], 
                    pywt.Wavelet(self.wavelet_filter), 
                    level=self.wavelet_level+factor, 
                    mode=self.wavelet_mode
                )
            
            elif x.shape[-1] < self.wavelet_size[-1]:
                factor = int(np.log2(self.wavelet_size[-1] // x.shape[-1]))
                # import pdb; pdb.set_trace()
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(
                    x[i, ...], 
                    pywt.Wavelet(self.wavelet_filter), 
                    level=self.wavelet_level-factor, 
                    mode=self.wavelet_mode
                )        
            else:
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(
                    x[i, ...], 
                    pywt.Wavelet(self.wavelet_filter), 
                    level=self.wavelet_level, 
                    mode=self.wavelet_mode
                )
            
            out_coeff = (
                torch.zeros(
                    self.out_channels,
                    x_coeff[0].shape[-3],
                    x_coeff[0].shape[-2],
                    x_coeff[0].shape[-1],
                    device=x_coeff[0].device
                ),
                *(
                    {key: torch.zeros(
                        self.out_channels,
                        val.shape[-3],
                        val.shape[-2],
                        val.shape[-1],
                        device=val.device
                    ) for key, val in x_dict.items()}
                    for x_dict in x_coeff[1:]
                )
            )
                
            D_COE0, H_COE0, W_COE0 = x_coeff[0].shape[-3], x_coeff[0].shape[-2], x_coeff[0].shape[-1]
            D_COE1, H_COE1, W_COE1 = x_coeff[0].shape[-3], x_coeff[0].shape[-2], x_coeff[0].shape[-1]
            
            modes1_coe0 = min(self.modes1, D_COE0)
            modes2_coe0 = min(self.modes2, H_COE0)
            modes3_coe0 = min(self.modes3, W_COE0)
            modes1_coe1 = min(self.modes1, D_COE1)
            modes2_coe1 = min(self.modes2, H_COE1)
            modes3_coe1 = min(self.modes3, W_COE1)
            
            # Multiply relevant Wavelet modes
            tmp_aaa = x_coeff[0].clone()
            out_coeff[0][..., :modes1_coe0, :modes2_coe0, :modes3_coe0] = self.mul3d(
                tmp_aaa[..., :modes1_coe0, :modes2_coe0, :modes3_coe0], 
                self.weight[0][..., :modes1_coe0, :modes2_coe0, :modes3_coe0]
            )
            
            tmp_aad = x_coeff[1]['aad'].clone()
            out_coeff[1]['aad'][..., :modes1_coe1, :modes2_coe1, :modes3_coe1] = self.mul3d(
                tmp_aad[..., :modes1_coe1, :modes2_coe1, :modes3_coe1], self.weight[1][..., :modes1_coe1, :modes2_coe1, :modes3_coe1]
            )
            
            tmp_ada = x_coeff[1]['ada'].clone()
            out_coeff[1]['ada'][..., :modes1_coe1, :modes2_coe1, :modes3_coe1] = self.mul3d(
                tmp_ada[..., :modes1_coe1, :modes2_coe1, :modes3_coe1], self.weight[2][..., :modes1_coe1, :modes2_coe1, :modes3_coe1]
            )
            
            tmp_add = x_coeff[1]['add'].clone()
            out_coeff[1]['add'][..., :modes1_coe1, :modes2_coe1, :modes3_coe1] = self.mul3d(
                tmp_add[..., :modes1_coe1, :modes2_coe1, :modes3_coe1], 
                self.weight[3][..., :modes1_coe1, :modes2_coe1, :modes3_coe1]
            )
            
            tmp_daa = x_coeff[1]['daa'].clone()
            out_coeff[1]['daa'][..., :modes1_coe1, :modes2_coe1, :modes3_coe1] = self.mul3d(
                tmp_daa[..., :modes1_coe1, :modes2_coe1, :modes3_coe1],
                self.weight[4][..., :modes1_coe1, :modes2_coe1, :modes3_coe1]
            )
            
            tmp_dad = x_coeff[1]['dad'].clone()
            out_coeff[1]['dad'][..., :modes1_coe1, :modes2_coe1, :modes3_coe1] = self.mul3d(
                tmp_dad[..., :modes1_coe1, :modes2_coe1, :modes3_coe1],
                self.weight[5][..., :modes1_coe1, :modes2_coe1, :modes3_coe1]
            )
            
            tmp_dda = x_coeff[1]['dda'].clone()
            out_coeff[1]['dda'][..., :modes1_coe1, :modes2_coe1, :modes3_coe1] = self.mul3d(
                tmp_dda[..., :modes1_coe1, :modes2_coe1, :modes3_coe1],
                self.weight[6]
            )
            
            tmp_ddd = x_coeff[1]['ddd'].clone()
            out_coeff[1]['ddd'][..., :modes1_coe1, :modes2_coe1, :modes3_coe1] = self.mul3d(
                tmp_ddd[..., :modes1_coe1, :modes2_coe1, :modes3_coe1],
                self.weight[7][..., :modes1_coe1, :modes2_coe1, :modes3_coe1]
            )
            
            # Instantiate higher wavelet_level coefficients as zeros
            # for jj in range(2, self.wavelet_level + 1):
            #     out_coeff[jj] = {key: torch.zeros([*x_coeff[jj][key].shape], device=x.device)
            #                     for key in out_coeff[jj].keys()}

            # Return to physical space        
            xr[i, ...] = waverec3(out_coeff, pywt.Wavelet(self.wavelet_filter))
        return xr
