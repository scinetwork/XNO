import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import time


# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================  

class PR3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(PR3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_pole3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes3, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, self.modes3, dtype=torch.cfloat))

    def output_PR(self, lambda1, lambda2, lambda3, alpha, weights_pole1, weights_pole2, weights_pole3, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],weights_residue.shape[4],lambda1.shape[0], lambda2.shape[0], lambda2.shape[3], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik,rbio->pqrbixko",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2),torch.sub(lambda3,weights_pole3)))
        Hw=torch.einsum("bixko,pqrbixko->pqrbixko",weights_residue,term1)
        output_residue1=torch.einsum("bioxs,oxsikpqr->bkoxs", alpha, Hw) 
        output_residue2=torch.einsum("bioxs,oxsikpqr->bkpqr", alpha, -Hw) 
        return output_residue1,output_residue2
    

    def forward(self, x):
        # tt=T.cuda()
        # tx=X.cuda()
        # ty=Y.cuda()
        # #Compute input poles and resudes by FFT
        # dty=(ty[0,1]-ty[0,0]).item()  # location interval
        # dtx=(tx[0,1]-tx[0,0]).item()  # location interval
        # dtt=(tt[0,1]-tt[0,0]).item()  # time interval
        
        nt = 39  # Number of time steps
        nx = 28  # Original spatial resolution
        ny = 28
        r = 2    # Reduction factor

        # Reduced resolution
        nx_reduced = int(((nx - 1) / r) + 1)  # nx' = 15
        ny_reduced = int(((ny - 1) / r) + 1)  # ny' = 15

        # Time grid
        T = torch.linspace(0, nt - 1, nt).reshape(1, nt).to(device)  # Shape: (1, nt)

        # Spatial grids
        X = torch.linspace(0, 1, steps=nx).reshape(1, nx)[:, :nx_reduced].to(device)  # Shape: (1, nx')
        Y = torch.linspace(0, 1, steps=ny).reshape(1, ny)[:, :ny_reduced].to(device)  # Shape: (1, ny')
        
        tt = T
        tx = X
        ty = Y
        # Spacings
        dtt = (T[0, 1] - T[0, 0]).item()  # Time interval
        dtx = (X[0, 1] - X[0, 0]).item()  # X spatial interval
        dty = (Y[0, 1] - Y[0, 0]).item()  # Y spatial interval

        
        
        alpha = torch.fft.fftn(x, dim=[-3,-2,-1])
        omega1=torch.fft.fftfreq(tt.shape[1], dtt)*2*np.pi*1j   # time frequency
        omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # location frequency
        omega3=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # location frequency
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
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, tt.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bim,ky->bimy", self.weights_pole3, ty.type(torch.complex64).reshape(1,-1))
        term4=torch.einsum("bipz,biqx,bimy->bipqmzxy", torch.exp(term1),torch.exp(term2),torch.exp(term3))
        x2=torch.einsum("kbpqm,bipqmzxy->kizxy", output_residue2,term4)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)/x.size(-3)
        return x1+x2
