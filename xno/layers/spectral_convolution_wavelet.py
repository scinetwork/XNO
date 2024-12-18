import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple, Union
from .resample import resample


Number = Union[int, float]

try:
    import ptwt, pywt
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT 
except ImportError:
    raise ImportError("Required packages are missing. Please install dependencies as per the documentation.")


""" Def: 1d Wavelet convolutional layer """
class SpectralConvWavelet1D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level, 
        wavelet_size, 
        wavelet=['db4'],
        wavelet_mode='symmetric',
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
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
        wavelet_size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter
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
        if np.isscalar(wavelet_size):
            self.wavelet_size = wavelet_size
        else:
            raise Exception("wavelet_size: WaveConv1d accepts signal length in scalar only") 
        self.wavelet = wavelet[0]
        self.wavelet_mode = wavelet_mode
        self.resolution_scaling_factor = resolution_scaling_factor
        
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.wavelet_level, mode=self.wavelet_mode)
        dummy_data = torch.randn( 1,1,self.wavelet_size ) 
        mode_data, _ = self.dwt_(dummy_data)
        self.modes1 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
    
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
        if x.shape[-1] > self.wavelet_size:
            factor = int(np.log2(x.shape[-1] // self.wavelet_size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.wavelet_level+factor, mode=self.wavelet_mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.wavelet_size:
            factor = int(np.log2(self.wavelet_size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.wavelet_level-factor, mode=self.wavelet_mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.wavelet_level, mode=self.wavelet_mode).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final low pass wavelet coefficients
        out_ft = self.mul1d(x_ft, self.weights1)
        # Multiply the final high pass wavelet coefficients
        out_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2)
    
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.wavelet_mode).to(x.device)
        x = idwt((out_ft, out_coeff)) 
        return x


""" Def: 2d Wavelet convolutional layer (discrete) """
class SpectralConvWavelet2D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level, 
        wavelet_size, 
        wavelet,
        wavelet_mode='symmetric', 
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
    ):
        super(SpectralConvWavelet2D, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        wavelet_level        : scalar, levels of wavelet decomposition
        wavelet_size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filters
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
                raise Exception('wavelet_size: WaveConv2dCwt accepts the wavelet_size of 2D signal in list with 2 elements')
            else:
                self.wavelet_size = wavelet_size
        else:
            raise Exception('wavelet_size: WaveConv2dCwt accepts wavelet_size of 2D signal is list')
        self.wavelet = wavelet[0]       
        self.wavelet_mode = wavelet_mode
        self.resolution_scaling_factor = resolution_scaling_factor
        
        dummy_data = torch.randn( 1,1,*self.wavelet_size )        
        dwt_ = DWT(J=self.wavelet_level, mode=self.wavelet_mode, wave=self.wavelet)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        
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
        if x.shape[-1] > self.wavelet_size[-1]:
            factor = int(np.log2(x.shape[-1] // self.wavelet_size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.wavelet_level+factor, mode=self.wavelet_mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.wavelet_size[-1]:
            factor = int(np.log2(self.wavelet_size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.wavelet_level-factor, mode=self.wavelet_mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.wavelet_level, mode=self.wavelet_mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode=self.wavelet_mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))
        return x

    
""" Def: 2d Wavelet convolutional layer (slim continuous) """
class SpectralConvWavelet2DCwt(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level, 
        wavelet_size,
        wavelet=['near_sym_b', 'qshift_b'],
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None
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
        wavelet_size         : scalar, length of input 1D signal
        wavelet1     : string, Specifies the first level biorthogonal wavelet filters
        wavelet2     : string, Specifies the second level quarter shift filters
        wavelet_mode         : string, padding style for wavelet decomposition
        
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
        self.wavelet_level1 = wavelet[0]
        self.wavelet_level2 = wavelet[1]       
        self.resolution_scaling_factor = resolution_scaling_factor
        
        dummy_data = torch.randn( 1,1,*self.wavelet_size ) 
        dwt_ = DTCWTForward(J=self.wavelet_level, biort=self.wavelet_level1, qshift=self.wavelet_level2)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights15r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights15c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        
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
        if x.shape[-1] > self.wavelet_size[-1]:
            factor = int(np.log2(x.shape[-1] // self.wavelet_size[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.wavelet_level+factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
            
        elif x.shape[-1] < self.wavelet_size[-1]:
            factor = int(np.log2(self.wavelet_size[-1] // x.shape[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.wavelet_level-factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)            
        else:
            # Compute dual tree continuous Wavelet coefficients 
            cwt = DTCWTForward(J=self.wavelet_level, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights0)
        # Multiply the final detailed wavelet coefficients        
        out_coeff[-1][:,:,0,:,:,0] = self.mul2d(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights15r)
        out_coeff[-1][:,:,0,:,:,1] = self.mul2d(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights15c)
        out_coeff[-1][:,:,1,:,:,0] = self.mul2d(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights45r)
        out_coeff[-1][:,:,1,:,:,1] = self.mul2d(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights45c)
        out_coeff[-1][:,:,2,:,:,0] = self.mul2d(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights75r)
        out_coeff[-1][:,:,2,:,:,1] = self.mul2d(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights75c)
        out_coeff[-1][:,:,3,:,:,0] = self.mul2d(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights105r)
        out_coeff[-1][:,:,3,:,:,1] = self.mul2d(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights105c)
        out_coeff[-1][:,:,4,:,:,0] = self.mul2d(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights135r)
        out_coeff[-1][:,:,4,:,:,1] = self.mul2d(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights135c)
        out_coeff[-1][:,:,5,:,:,0] = self.mul2d(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights165r)
        out_coeff[-1][:,:,5,:,:,1] = self.mul2d(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights165c)        
        
        # Reconstruct the signal
        icwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
        x = icwt((out_ft, out_coeff))
        return x
    
    
""" Def: 3d Wavelet convolutional layer """
class SpectralConvWavelet3D(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        wavelet_level, 
        wavelet_size, 
        wavelet=['db4'], 
        wavelet_mode='periodic',
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
    ):
        super(SpectralConvWavelet3D, self).__init__()

        """
        3D Wavelet layer. It does 3D DWT, linear transform, and Inverse dWT.    
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        wavelet_level        : scalar, levels of wavelet decomposition
        wavelet_size         : scalar, length of input 1D signal
        wavelet      : string, Specifies the first level biorthogonal wavelet filters
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
                raise Exception('wavelet_size: WaveConv2dCwt accepts the wavelet_size of 3D signal in list with 3 elements')
            else:
                self.wavelet_size = wavelet_size
        else:
            raise Exception('wavelet_size: WaveConv2dCwt accepts wavelet_size of 3D signal is list')
        self.wavelet = wavelet[0]
        self.mwavelet_modeode = wavelet_mode
        dummy_data = torch.randn( [*self.wavelet_size] ).unsqueeze(0)
        mode_data = wavedec3(dummy_data, pywt.Wavelet(self.wavelet), level=self.wavelet_level, mode=self.wavelet_mode)
        self.modes1 = mode_data[0].shape[-3]
        self.modes2 = mode_data[0].shape[-2]
        self.modes3 = mode_data[0].shape[-1]
        self.resolution_scaling_factor = resolution_scaling_factor
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
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
        xr = torch.zemusoros(x.shape, device = x.device)
        for i in range(x.shape[0]):
            
            if x.shape[-1] > self.wavelet_size[-1]:
                factor = int(np.log2(x.shape[-1] // self.wavelet_size[-1]))
                
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(x[i, ...], pywt.Wavelet(self.wavelet), level=self.wavelet_level+factor, mode=self.wavelet_mode)
            
            elif x.shape[-1] < self.wavelet_size[-1]:
                factor = int(np.log2(self.wavelet_size[-1] // x.shape[-1]))
                
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(x[i, ...], pywt.Wavelet(self.wavelet), level=self.wavelet_level-factor, mode=self.wavelet_mode)        
            else:
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(x[i, ...], pywt.Wavelet(self.wavelet), level=self.wavelet_level, mode=self.wavelet_mode)
            
            # Multiply relevant Wavelet modes
            x_coeff[0] = self.mul3d(x_coeff[0].clone(), self.weights1)
            x_coeff[1]['aad'] = self.mul3d(x_coeff[1]['aad'].clone(), self.weights2)
            x_coeff[1]['ada'] = self.mul3d(x_coeff[1]['ada'].clone(), self.weights3)
            x_coeff[1]['add'] = self.mul3d(x_coeff[1]['add'].clone(), self.weights4)
            x_coeff[1]['daa'] = self.mul3d(x_coeff[1]['daa'].clone(), self.weights5)
            x_coeff[1]['dad'] = self.mul3d(x_coeff[1]['dad'].clone(), self.weights6)
            x_coeff[1]['dda'] = self.mul3d(x_coeff[1]['dda'].clone(), self.weights7)
            x_coeff[1]['ddd'] = self.mul3d(x_coeff[1]['ddd'].clone(), self.weights8)
            
            # Instantiate higher wavelet_level coefficients as zeros
            for jj in range(2, self.wavelet_level + 1):
                x_coeff[jj] = {key: torch.zeros([*x_coeff[jj][key].shape], device=x.device)
                                for key in x_coeff[jj].keys()}
            
            # Return to physical space        
            xr[i, ...] = waverec3(x_coeff, pywt.Wavelet(self.wavelet))
        return xr
