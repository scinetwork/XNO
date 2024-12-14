import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Sequence
from typing import List, Optional, Tuple, Union
from .resample import resample


Number = Union[int, float]


def _compute_dt(shape, start_points=None, end_points=None):
    """
    Compute uniform spacing (dt) for each dimension based on domain lengths, step sizes,
    start points, and end points. Defaults to a unit domain if not specified.

    Parameters:
    shape (Sequence[int]): The shape of the input excluding batch and channel, i.e. (d_1, d_2, ..., d_n).
    step_sizes (Sequence[float], optional): Step sizes for each dimension. Defaults to shape-based uniform spacing.
    start_points (Sequence[float], optional): Start points for each dimension. Defaults to 0.0 for all dimensions.
    end_points (Sequence[float], optional): End points for each dimension. Defaults to 1.0 for all dimensions.

    Returns:
    dt_list (Sequence[float]): A list of spacings, one per dimension.
    grid (List[torch.Tensor]): A list of grid points for each dimension based on the spacing and domain.
    """
    dim = len(shape)

    # Set default start and end points if not provided
    if start_points is None:
        start_points = torch.zeros(dim).tolist()
    if end_points is None:
        end_points = torch.ones(dim).tolist()

    # Validate that start_points and end_points match the number of dimensions
    if len(start_points) != dim or len(end_points) != dim:
        raise ValueError("Start points and end points must match the number of input dimensions ({dim}).")

    # Compute domain lengths from start and end points
    domain_lengths = [end_points[i] - start_points[i] for i in range(dim)]

    # Generate grid points for each dimension using torch.linspace
    grid = [torch.linspace(start_points[i], end_points[i], steps=shape[i]) for i in range(dim)]

    # Compute dt directly from the grid
    dt_list = [(grid[i][1] - grid[i][0]).item() for i in range(dim)]

    return dt_list, grid



# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================
class SpectralConvLaplace1D(nn.Module):
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
        linspace_steps=None, 
        linspace_startpoints=None, 
        linspace_endpoints=None, 
        
        ):
        super(SpectralConvLaplace1D, self).__init__()
        
        
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        
        self.resolution_scaling_factor = resolution_scaling_factor
        
        modes = list(n_modes)
        self.modes1 = modes[0]
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
       
    
    def transform(self, x, output_shape=None):
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
    
    def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.sub(lambda1,weights_pole))
        Hw=weights_residue*term1
        output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2=torch.einsum("bix,xiok->bok", alpha, -Hw) 
        return output_residue1,output_residue2    

    def forward(
        self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None
    ):
        
        # t=grid_x_train
        # #Compute input poles and resudes by FFT
        # dt=(t[1]-t[0]).item()
        
        if self.linspace_steps is None:
            self.linspace_steps = x.shape[2:]
            
        dt_list, shape = _compute_dt(shape=self.linspace_steps, 
                                     start_points=self.linspace_startpoints, 
                                     end_points=self.linspace_endpoints)
        t = shape[0]
        dt = dt_list[0]        
        
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1
    
    
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)
    
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1) 
        x2=torch.einsum("bix,ioxz->boz", output_residue2,term2)
        x2=torch.real(x2)
        x2=x2/x.size(-1)
        return x1+x2


class SpectralConvLaplace2D(nn.Module):
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
        linspace_steps=None, 
        linspace_startpoints=None, 
        linspace_endpoints=None
        ):
        
        super(SpectralConvLaplace2D, self).__init__()
        
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        
        self.resolution_scaling_factor = resolution_scaling_factor
        
        # modes = list(n_modes)
        # self.modes1 = modes[0]
        # self.modes2 = modes[1]
        # self.scale = (1 / (in_channels*out_channels))
        # self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        # self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        # self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, dtype=torch.cfloat))
        
        """
            Commulating all weights into a single weight attribute on the class, and break it down for different applications like weights_pole1, etc. in convolution process. 
        """
        # Handle n_modes and max_n_modes
        self.n_modes = n_modes  # Uses the setter
        if max_n_modes is None:
            self.max_n_modes = self.n_modes
        else:
            self.max_n_modes = max_n_modes
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Initialize single weight tensor combining poles and residues
        # Shape: (in_channels, out_channels, modes1 + modes2 + modes1 * modes2)
        if isinstance(self.max_n_modes, int):
            max_modes1 = max_modes2 = self.max_n_modes
        else:
            max_modes1, max_modes2 = self.max_n_modes
        
        total_modes = max_modes1 + max_modes2 + (max_modes1 * max_modes2)
        self.weight = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, total_modes, dtype=torch.cfloat)
        )
        
        
    def transform(self, x, output_shape=None):
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
    
    def output_PR(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik->pqbixk",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2)))
        Hw=torch.einsum("bixk,pqbixk->pqbixk",weights_residue,term1)
        Pk=Hw  # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw; 
        
        output_residue1=torch.einsum("biox,oxikpq->bkox", alpha, Hw) 
        output_residue2=torch.einsum("biox,oxikpq->bkpq", alpha, Pk) 
        return output_residue1,output_residue2

    # def forward(
    #     self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None
    # ):
    #     """
    #         Dynamically assigning different shapes of the self.weight to the each set of required weights.
    #         This is dynamic because of the grad_explained() in incremental.py. 
    #     """
    #     modes1, modes2 = self.n_modes
    #     start_pole1 = 0
    #     end_pole1 = modes1
    #     start_pole2 = end_pole1
    #     end_pole2 = start_pole2 + modes2
    #     start_residue = end_pole2
    #     end_residue = start_residue + (modes1 * modes2)

    #     # Pre-slice weights
    #     self.weights_pole1 = self.weight[:, :, start_pole1:end_pole1].view(self.weight.size(0), self.weight.size(1), modes1)
    #     self.weights_pole2 = self.weight[:, :, start_pole2:end_pole2].view(self.weight.size(0), self.weight.size(1), modes2)
    #     self.weights_residue = self.weight[:, :, start_residue:end_residue].view(self.weight.size(0), self.weight.size(1), modes1, modes2)
        
    #     # tx=T
    #     # ty=X
    #     # #Compute input poles and resudes by FFT
    #     # dty=(ty[0,1]-ty[0,0]).item()  # location interval
    #     # dtx=(tx[0,1]-tx[0,0]).item()  # time interval
        
    #     if self.linspace_steps is None:
    #         self.linspace_steps = x.shape[2:]
            
    #     dt_list, shape = _compute_dt(shape=self.linspace_steps, 
    #                                  start_points=self.linspace_startpoints, 
    #                                  end_points=self.linspace_endpoints)
        
    #     print(f"X shape: {x.shape}")
    #     ty = shape[0]
    #     tx = shape[1]
    #     dty = dt_list[0] 
    #     dtx = dt_list[1] 
    #     alpha = torch.fft.fft2(x, dim=[-2,-1])
    #     omega1=torch.fft.fftfreq(ty.shape[0], dty)*2*np.pi*1j   # location frequency
    #     omega2=torch.fft.fftfreq(tx.shape[0], dtx)*2*np.pi*1j   # time frequency
    #     omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #     omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #     lambda1=omega1
    #     lambda2=omega2
 
    #     # Obtain output poles and residues for transient part and steady-state part
    #     output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)

    #     # Obtain time histories of transient response and steady-state response
    #     x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
    #     x1 = torch.real(x1)    
    #     term1=torch.einsum("bip,kz->bipz", self.weights_pole1, ty.type(torch.complex64).reshape(1,-1))
    #     term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
    #     term3=torch.einsum("bipz,biqx->bipqzx", torch.exp(term1),torch.exp(term2))
    #     x2=torch.einsum("kbpq,bipqzx->kizx", output_residue2,term3)
    #     x2=torch.real(x2)
    #     x2=x2/x.size(-1)/x.size(-2)
    #     return x1+x2
    
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        modes1, modes2 = self.n_modes
        H, W = x.shape[-2], x.shape[-1]

        # Ensure we do not exceed the actual resolution
        modes1 = min(modes1, H)
        modes2 = min(modes2, W)
    
        
        # if self.linspace_steps is None:
        #     self.linspace_steps = x.shape[2:]
            
        self.linspace_steps = x.shape[2:]

        dt_list, shape = _compute_dt(shape=self.linspace_steps, 
                                    start_points=self.linspace_startpoints, 
                                    end_points=self.linspace_endpoints)

        ty = shape[0]
        tx = shape[1]
        dty = dt_list[0]
        dtx = dt_list[1]
                
        alpha = torch.fft.fft2(x, dim=[-2, -1])

        # Compute frequency grids
        omega1 = torch.fft.fftfreq(ty.shape[0], dty)*2*np.pi*1j
        omega2 = torch.fft.fftfreq(tx.shape[0], dtx)*2*np.pi*1j

        # Slice frequencies to match the chosen modes
        omega1 = omega1[:modes1]
        omega2 = omega2[:modes2]

        omega1 = omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2 = omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1 = omega1
        lambda2 = omega2

        # Also slice alpha to only consider the selected modes
        alpha = alpha[:, :, :modes1, :modes2]

        # Slice weights to match the truncated modes
        self.weights_pole1 = self.weight[:, :, :modes1].view(self.weight.size(0), self.weight.size(1), modes1)
        self.weights_pole2 = self.weight[:, :, modes1:(modes1+modes2)].view(self.weight.size(0), self.weight.size(1), modes2)
        self.weights_residue = self.weight[:, :, (modes1+modes2):(modes1+modes2+modes1*modes2)].view(self.weight.size(0), self.weight.size(1), modes1, modes2)

        # Proceed with the existing logic
        output_residue1, output_residue2 = self.output_PR(lambda1, lambda2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)
        
        x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
        x1 = torch.real(x1)  # shape: (b, o, H, W)

        # Let's say you have only one input channel or you just take mean over i:
        # weights_pole1: [i, o, p], weights_pole2: [i, o, q]
        # If multiple input channels: average over i (or select i=0)
        # weights_pole1_single = self.weights_pole1.mean(dim=0)  # now (o, p)
        # weights_pole2_single = self.weights_pole2.mean(dim=0)  # now (o, q)
        
        # Resample ty and tx to match H and W
        # ty1 = torch.linspace(ty[0].item(), ty[-1].item(), H, device=ty.device, dtype=ty.dtype)
        # tx1 = torch.linspace(tx[0].item(), tx[-1].item(), W, device=tx.device, dtype=tx.dtype)

        # Now ty has length H and tx has length W
        term1 = torch.einsum("iop,z->iopz", self.weights_pole1, ty.type(torch.complex64))  # (i, o, p, H)
        term2 = torch.einsum("ioq,x->ioqx", self.weights_pole2, tx.type(torch.complex64))  # (i, o, q, W)        

        term1 = torch.exp(term1)  # (i, o, p, H)
        term2 = torch.exp(term2)  # (i, o, q, W)

        term3 = torch.einsum("iopz,ioqx->iopqzx", term1, term2)  # (i, o, p, q, H, W)

        # output_residue2: (b, o, p, q)
        # term3: (o, p, q, H, W)
        x2 = torch.einsum("bopq,iopqzx->bozx", output_residue2, term3)  # (b, o, H, W)

        x2 = torch.real(x2) / (x.size(-1) * x.size(-2))

        return x1 + x2  # Both are (b, o, H, W)


class SpectralConvLaplace3D(nn.Module):
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
        linspace_steps=None, 
        linspace_startpoints=None, 
        linspace_endpoints=None
        ):
        super(SpectralConvLaplace3D, self).__init__()
        
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        
        self.resolution_scaling_factor = resolution_scaling_factor
        
        modes = list(n_modes)

        self.modes1 = modes[0]
        self.modes2 = modes[1]
        self.modes3 = modes[2]
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_pole3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes3, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, self.modes3, dtype=torch.cfloat))

    
    def transform(self, x, output_shape=None):
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
    
    
    def output_PR(self, lambda1, lambda2, lambda3, alpha, weights_pole1, weights_pole2, weights_pole3, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],weights_residue.shape[4],lambda1.shape[0], lambda2.shape[0], lambda2.shape[3], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik,rbio->pqrbixko",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2),torch.sub(lambda3,weights_pole3)))
        Hw=torch.einsum("bixko,pqrbixko->pqrbixko",weights_residue,term1)
        output_residue1=torch.einsum("bioxs,oxsikpqr->bkoxs", alpha, Hw) 
        output_residue2=torch.einsum("bioxs,oxsikpqr->bkpqr", alpha, -Hw) 
        return output_residue1,output_residue2
    

    def forward(
        self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None
    ):
        # tt=T
        # tx=X
        # ty=Y
        
        # dty=(ty[0,1]-ty[0,0]).item()  # location interval
        # dtx=(tx[0,1]-tx[0,0]).item()  # location interval
        # dtt=(tt[0,1]-tt[0,0]).item()  # time interval
        
        if self.linspace_steps is None:
            self.linspace_steps = x.shape[2:]
            
        dt_list, shape = _compute_dt(shape=self.linspace_steps, 
                                     start_points=self.linspace_startpoints, 
                                     end_points=self.linspace_endpoints)
        tz = shape[0]
        ty = shape[1]
        tx = shape[2]
        # #Compute input poles and resudes by FFT
        dtz = dt_list[0] # this can be time dimension, instead of Z dimension
        dty = dt_list[1]
        dtx = dt_list[2] 
        
        alpha = torch.fft.fftn(x, dim=[-3,-2,-1])
        omega1=torch.fft.fftfreq(tz.shape[0], dtz)*2*np.pi*1j   # time frequency
        omega2=torch.fft.fftfreq(tx.shape[0], dtx)*2*np.pi*1j   # location frequency
        omega3=torch.fft.fftfreq(ty.shape[0], dty)*2*np.pi*1j   # location frequency
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega3=omega3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=omega1
        lambda2=omega2    
        lambda3=omega3

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, lambda3, alpha, self.weights_pole1, self.weights_pole2, self.weights_pole3, self.weights_residue)
 
      
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifftn(output_residue1, s=(x.size(-3),x.size(-2), x.size(-1)))
        x1 = torch.real(x1)
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, tz.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bim,ky->bimy", self.weights_pole3, ty.type(torch.complex64).reshape(1,-1))
        term4=torch.einsum("bipz,biqx,bimy->bipqmzxy", torch.exp(term1),torch.exp(term2),torch.exp(term3))
        x2=torch.einsum("kbpqm,bipqmzxy->kizxy", output_residue2,term4)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)/x.size(-3)
        return x1+x2