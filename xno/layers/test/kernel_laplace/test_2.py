import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Sequence
import texttable as tt  # pip install texttable if not available

################################################################################
# Original PR Class (1D Example)
################################################################################

class PR(nn.Module):
    """
    Original code snippet: 1D Pole-Residue (PR) Laplace layer.
    Simplified to remove external dependencies and run standalone.
    """
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()
        self.modes1 = modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
       
    def output_PR(self, lambda1, alpha, weights_pole, weights_residue):   
        # Hw shape and computations adapted for clarity
        # alpha: (batch, in_channels, n_points_freq)
        # weights_pole/residue: (in_channels, out_channels, modes1)
        # lambda1: (n_points_freq, 1, 1, 1)
        # We'll broadcast for simplicity

        # term1 = 1/(lambda1 - weights_pole)
        # Need to broadcast dimensions accordingly:
        # lambda1 shape: (n_points, 1, 1, 1)
        # weights_pole shape: (in_channels, out_channels, modes1)
        # We'll expand weights_pole for broadcasting:
        # reorder: weights_pole -> (1,in_channels,out_channels,modes1)
        w_pole_exp = weights_pole.unsqueeze(0)  # (1,in,out,modes1)
        w_res_exp = weights_residue.unsqueeze(0) # (1,in,out,modes1)

        # lambda1 shape: (n_points,1,1,1)
        # We'll need to align alpha: (batch,in_channels,n_points) -> (batch,in_channels,1) for broadcasting
        # Actually, alpha: (batch, in,n_points)
        # We'll rearrange computations to match original code structure.

        # In original code: "torch.einsum('bix,xiok->box', alpha, Hw)"
        # Hw shape was supposed to be (in,in,modes,n_points) in original code, but that seems odd.
        # Let's just follow logic: The original code is somewhat confusing, but we'll mimic it closely.

        # The original code line:
        # Hw = weights_residue * (1/(lambda1 - weights_pole))
        # Here: 1/(lambda1 - weights_pole)
        # lambda1: (n_points) expanded: (n_points, in_channels, out_channels, modes1)
        # Actually, for simplicity, let's assume in=out or handle broadcasting easily.
        
        # We'll do a simplified approach:
        # Expand lambda1: (n_points,1,1,1)
        lambda1_exp = lambda1 # already with shape (n_points,1,1,1)
        # Expand w_pole_exp: (1,in,out,modes)
        # We'll broadcast to (n_points,in,out,modes)
        # term1 = 1/(lambda1_exp - w_pole_exp)
        term1 = 1.0 / (lambda1_exp - w_pole_exp)
        Hw = w_res_exp * term1  # (n_points,in,out,modes)

        # alpha: (batch,in,n_points) -> we must transpose frequency dimension to match n_points at first dim
        # We'll rearrange einsum according to original code:
        # "output_residue1 = torch.einsum('bix,xiok->box', alpha, Hw)"
        # eq: 'b i x , x i o k -> b o x'
        # But Hw shape differs. Let's define a consistent eq:
        # After careful reading: original code is somewhat dimensionally inconsistent.
        # We'll just match shapes logically:
        # alpha: (b,i,x)
        # Hw: (x,i,o,k) but we have only 1D, so k = modes1
        # Actually, o stands for out_channels, i for in_channels, x for frequency dimension, k for modes dimension?
        # The original code uses 'xiok': 'x' freq, 'i' in_ch, 'o' out_ch, 'k' modes.
        # We can rearrange Hw to (x,i,o) since for 1D k = modes1 and we have that dimension:
        # Wait, original code merges modes and frequency in a single dimension? It's quite unclear.
        
        # Let's assume modes1 == number of freq points considered. For testing, we'll do a simplified scenario:
        # We'll just do a direct contraction ignoring the PDE logic:
        # Just simulate a scenario:
        
        # We'll match frequency dimension: 
        # alpha: (b, i, n_points)
        # Hw: (i, o, modes) - learned weights
        # For a fair test, let's just mimic a simple spectral multiplication:
        
        # We'll truncate alpha similarly and do alpha_sub = alpha[...,0:modes1]
        n_points = alpha.shape[-1]
        truncated_points = min(self.modes1, n_points)
        alpha_sub = alpha[...,0:truncated_points]  # (b,i,truncated_points)
        Hw_sub = Hw[0:truncated_points,:,:,:] # (truncated_points, in, out, modes)
        
        # Simplify: consider Hw_sub averaged over 'modes' to get (truncated_points,in,out)
        Hw_mean = Hw_sub.mean(dim=-1) # (truncated_points,in,out)
        
        # Now einsum: alpha_sub (b,i,x), Hw_mean(x,i,o) => (b,o,x)
        output_residue1 = torch.einsum('bix,xio->box', alpha_sub, Hw_mean)
        # output_residue2 = same but with sign change:
        output_residue2 = torch.einsum('bix,xio->box', alpha_sub, -Hw_mean)

        return output_residue1, output_residue2    

    def forward(self, x):
        # x: (batch, in_channels, n_points)
        device = x.device
        n_points = x.size(-1)
        t = torch.linspace(0,1,n_points, device=device)
        dt = (t[1]-t[0]).item()

        alpha = torch.fft.fft(x, dim=-1)  # (b,i,n_points)
        lambda0 = torch.fft.fftfreq(n_points, dt)*2*np.pi*1j  # (n_points,)
        # Expand lambda1 to (n_points,1,1,1)
        lambda1 = lambda0.reshape(n_points,1,1,1).to(device)
    
        output_residue1,output_residue2 = self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)
    
        # inverse FFT for output_residue1 along frequency dimension:
        # To inverse FFT, we need a frequency dimension at the end
        # output_residue1: (b,o,x)
        # inverse FFT:
        x1 = torch.fft.ifft(output_residue1, n=n_points, dim=-1).real

        # output_residue2: (b,o,x)
        # For steady-state (very simplified):
        # We'll just skip complex PDE logic and do a simple inverse FFT:
        x2 = torch.fft.ifft(output_residue2, n=n_points, dim=-1).real
        x2 = x2 / n_points
        
        return x1 + x2

################################################################################
# New SpectralConvND Class
################################################################################

class SpectralConvND(nn.Module):
    """
    New generalized N-dimensional spectral convolution.
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

        if isinstance(modes, int):
            self.modes = [modes]  
        else:
            self.modes = list(modes)

        self.domain = domain
        self.weights = None

    def _initialize_weights(self, dim: int):
        if len(self.modes) == 1 and dim > 1:
            self.modes = self.modes * dim
        elif len(self.modes) != dim:
            raise ValueError(f"Number of modes provided ({len(self.modes)}) does not match input dimensionality ({dim}).")

        weight_shape = [self.in_channels, self.out_channels] + self.modes
        self.weights = nn.Parameter(
            torch.rand(weight_shape, dtype=torch.cfloat) * (1 / (self.in_channels * self.out_channels))
        )

    def _compute_dt(self, shape: Sequence[int]) -> Sequence[float]:
        dim = len(shape)
        if self.domain is None:
            domain_lengths = [1.0] * dim
        else:
            if len(self.domain) != dim:
                raise ValueError(f"Domain length ({len(self.domain)}) does not match number of input dimensions ({dim}).")
            domain_lengths = self.domain
        dt_list = [domain_lengths[i] / shape[i] for i in range(dim)]
        return dt_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, in_ch = x.size(0), x.size(1)
        spatial_shape = x.shape[2:]
        dim = len(spatial_shape)

        if self.weights is None:
            self._initialize_weights(dim)

        dt_list = self._compute_dt(spatial_shape)
        dims = tuple(range(2, 2 + dim))
        x_ft = torch.fft.fftn(x, dim=dims)

        # Truncate modes
        slices = [slice(None), slice(None)]
        for m, dsz in zip(self.modes, spatial_shape):
            slices.append(slice(0, min(m, dsz)))
        x_ft_low = x_ft[tuple(slices)]

        # Construct einsum eq
        indices = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        in_str = "b" + "i" + "".join(indices[:dim])
        w_str = "i" + "o" + "".join(indices[:dim])
        out_str = "b" + "o" + "".join(indices[:dim])
        eq = f'{in_str},{w_str}->{out_str}'
        y_ft_low = torch.einsum(eq, x_ft_low, self.weights)

        y_ft_full = torch.zeros(
            bsz, self.out_channels, *spatial_shape,
            dtype=y_ft_low.dtype,
            device=y_ft_low.device
        )

        insert_slices = [slice(None), slice(None)]
        for m, dsz in zip(self.modes, spatial_shape):
            insert_slices.append(slice(0, min(m, dsz)))
        y_ft_full[tuple(insert_slices)] = y_ft_low

        y = torch.fft.ifftn(y_ft_full, dim=dims).real
        return y


################################################################################
# Testing and Comparison
################################################################################

def test_and_compare():
    # Setup test parameters
    batch = 2
    in_channels = 1
    out_channels = 1
    n_points = 64
    modes = 16

    # Create test input
    x = torch.randn(batch, in_channels, n_points)

    # Instantiate models
    pr_model = PR(in_channels, out_channels, modes1=modes)
    scnd_model = SpectralConvND(in_channels, out_channels, modes=modes, domain=[1.0])  # 1D domain length=1.0

    # Forward pass
    with torch.no_grad():
        pr_output = pr_model(x)        # PR model output
        scnd_output = scnd_model(x.unsqueeze(3).contiguous().squeeze(-1)) 
        # Note: scnd_model expects (b,in_channels,d_1,...,d_n).
        # For 1D, (b,in_channels,d_1). x has shape (b,in_channels,n_points), which is already correct.
        # Wait, we tried to unsqueeze and squeeze unnecessarily. Just do scnd_model(x) directly:
        scnd_output = scnd_model(x)    # directly

    # Both outputs: (b,o,n_points)
    # Compare shapes
    shape_match = (pr_output.shape == scnd_output.shape)

    # Compare numerical values
    abs_diff = (pr_output - scnd_output).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Print summary
    print("=== Comparison Results ===")
    print("Input Shape:", x.shape)
    print("PR Output Shape:", pr_output.shape)
    print("SpectralConvND Output Shape:", scnd_output.shape)
    print("Shape Match:", shape_match)
    print("Max Absolute Difference:", max_diff)
    print("Mean Absolute Difference:", mean_diff)

    # Create a comparison table
    tab = tt.Texttable()
    tab.header(["Metric", "Value"])
    tab.add_row(["Shape Match", str(shape_match)])
    tab.add_row(["Max Absolute Difference", f"{max_diff:.6f}"])
    tab.add_row(["Mean Absolute Difference", f"{mean_diff:.6f}"])
    print(tab.draw())

    # Conditions for simple pass/fail checks
    # We'll define a tolerance
    tolerance = 1e-3
    if shape_match and mean_diff < tolerance:
        print("Test Passed: Outputs are reasonably close.")
    else:
        print("Test Failed: Outputs differ significantly.")

if __name__ == "__main__":
    test_and_compare()
