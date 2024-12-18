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
class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()

        self.modes1 = modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
        Pk=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.sub(lambda1,weights_pole))
        Hw=weights_residue*term1
        Pk=-Hw  
        output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2=torch.einsum("bix,xiok->bok", alpha, Pk) 
        return output_residue1,output_residue2    
    

    def forward(self, x):
        t=grid_x_train.cuda()
        #Compute input poles and resudes by FFT
        dt=(t[1]-t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.cuda()
        
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
