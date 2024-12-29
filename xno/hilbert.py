"""
@author: Saman Pordanesh
"""

"""
Hilbert transformation, compatible with the Pytorch tensors. 
The code has a test funciton for comparing the newly implemented Torch Hilbert transform, with Scipy package hilbert transform, calculating errors and plotting resulted transformation. 
"""

import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
import os

def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    Parameters
    ----------
    x : Tensor
        Signal data. Must be real.
    N : int, optional
        Number of Fourier components. Default: x.shape[axis]
    axis : int, optional
        Axis along which to do the transformation. Default: -1.

    Returns
    -------
    x : Tensor
        Analytic signal of x, along the specified axis.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.size(axis)
    if N <= 0:
        raise ValueError("N must be positive.")

    # Compute FFT along the specified axis
    Xf = torch.fft.fft(x, n=N, dim=axis)

    # Construct the filter
    h = torch.zeros(N, dtype=Xf.dtype, device=Xf.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    # Reshape h to broadcast along the correct axis
    shape = [1] * x.ndim
    shape[axis] = N
    h = h.view(shape)

    # Multiply Xf by h
    Xf = Xf * h

    # Compute inverse FFT
    x = torch.fft.ifft(Xf, n=None, dim=axis)

    return x

def hilbert2(x, N=None):
    """
    Compute the 2-D analytic signal of x along axes (0,1)

    Parameters
    ----------
    x : Tensor
        Signal data. Must be at least 2-D and real.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is x.shape[:2]

    Returns
    -------
    x : Tensor
        Analytic signal of x taken along axes (0,1).
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim < 2:
        raise ValueError("x must be at least 2-D.")
    if torch.is_complex(x):
        raise ValueError("x must be real.")

    if N is None:
        N = x.shape[:2]
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    elif len(N) != 2 or any(n <= 0 for n in N):
        raise ValueError("When given as a tuple, N must hold exactly two positive integers")

    # Compute 2D FFT along axes (0,1)
    Xf = torch.fft.fft2(x, s=N, dim=(0, 1))

    # Construct the filters
    h1 = torch.zeros(N[0], dtype=Xf.dtype, device=Xf.device)
    N0 = N[0]
    if N0 % 2 == 0:
        h1[0] = h1[N0 // 2] = 1
        h1[1:N0 // 2] = 2
    else:
        h1[0] = 1
        h1[1:(N0 + 1) // 2] = 2

    h2 = torch.zeros(N[1], dtype=Xf.dtype, device=Xf.device)
    N1 = N[1]
    if N1 % 2 == 0:
        h2[0] = h2[N1 // 2] = 1
        h2[1:N1 // 2] = 2
    else:
        h2[0] = 1
        h2[1:(N1 + 1) // 2] = 2

    # Construct the 2D filter h
    h = h1[:, None] * h2[None, :]

    # Expand h to match the dimensions of x
    h_shape = list(h.shape) + [1] * (x.ndim - 2)
    h = h.view(h_shape)

    # Multiply Xf by h
    Xf = Xf * h

    # Compute inverse FFT
    x = torch.fft.ifft2(Xf, s=None, dim=(0, 1))

    return x

# def run_tests():
#     # Create lists to store results
#     test_results = []
#     plot_index = 1

#     # Create a directory for plots
#     if not os.path.exists('plots'):
#         os.makedirs('plots')

#     # Define test scenarios
#     scenarios = [
#         # 1D Scenarios
#         {'type': '1D', 'function': np.sin, 'desc': 'Sine Wave'},
#         {'type': '1D', 'function': np.cos, 'desc': 'Cosine Wave'},
#         {'type': '1D', 'function': lambda t: np.sign(np.sin(t)), 'desc': 'Square Wave'},
#         {'type': '1D', 'function': lambda t: np.random.randn(len(t)), 'desc': 'Random Noise'},
#         {'type': '1D', 'function': lambda t: np.exp(-t**2), 'desc': 'Gaussian'},
#         # 2D Scenarios
#         {'type': '2D', 'function': lambda x, y: np.sin(x) + np.cos(y), 'desc': 'Sine + Cosine'},
#         {'type': '2D', 'function': lambda x, y: np.exp(-0.1*(x**2 + y**2)), 'desc': '2D Gaussian'},
#         {'type': '2D', 'function': lambda x, y: np.sign(np.sin(x) * np.sin(y)), 'desc': '2D Square Wave'},
#         {'type': '2D', 'function': lambda x, y: np.random.randn(*x.shape), 'desc': '2D Random Noise'},
#         {'type': '2D', 'function': lambda x, y: np.sin(5*x + 5*y), 'desc': 'High Frequency Sine'},
#     ]

#     for idx, scenario in enumerate(scenarios):
#         desc = scenario['desc']
#         if scenario['type'] == '1D':
#             # Generate test signal
#             t = np.linspace(0, 2 * np.pi, 500)
#             x_np = scenario['function'](t)

#             # Apply PyTorch Hilbert Transform
#             x_torch = torch.from_numpy(x_np)
#             analytic_torch = hilbert(x_torch)
#             hilbert_torch = analytic_torch.imag.numpy()

#             # Apply SciPy Hilbert Transform
#             analytic_scipy = scipy.signal.hilbert(x_np)
#             hilbert_scipy = np.imag(analytic_scipy)

#             # Calculate Error
#             error = np.abs(hilbert_torch - hilbert_scipy)
#             max_error = np.max(error)
#             mean_error = np.mean(error)

#             # Plot Results
#             plt.figure(figsize=(10, 6))
#             plt.plot(t, hilbert_scipy, label='SciPy Hilbert', alpha=0.7)
#             plt.plot(t, hilbert_torch, '--', label='PyTorch Hilbert', alpha=0.7)
#             plt.title(f'1D Hilbert Transform - {desc}')
#             plt.xlabel('Time')
#             plt.ylabel('Hilbert Transform')
#             plt.legend()
#             plt.savefig(f'plots/plot_{plot_index}.png')
#             plt.close()
#             plot_index += 1

#             # Store Results
#             test_results.append({
#                 'Scenario': f'1D - {desc}',
#                 'Max Error': max_error,
#                 'Mean Error': mean_error
#             })

#         elif scenario['type'] == '2D':
#             # Generate test signal
#             x = np.linspace(-5, 5, 100)
#             y = np.linspace(-5, 5, 100)
#             X, Y = np.meshgrid(x, y)
#             Z_np = scenario['function'](X, Y)

#             # Apply PyTorch Hilbert2 Transform
#             Z_torch = torch.from_numpy(Z_np)
#             analytic_torch = hilbert2(Z_torch)
#             hilbert_torch = analytic_torch.imag.numpy()

#             # Apply SciPy Hilbert2 Transform
#             analytic_scipy = scipy.signal.hilbert2(Z_np)
#             hilbert_scipy = np.imag(analytic_scipy)

#             # Calculate Error
#             error = np.abs(hilbert_torch - hilbert_scipy)
#             max_error = np.max(error)
#             mean_error = np.mean(error)

#             # Plot Results (Display a slice or the central row)
#             plt.figure(figsize=(10, 6))
#             idx_slice = Z_np.shape[0] // 2
#             plt.plot(x, hilbert_scipy[idx_slice, :], label='SciPy Hilbert2', alpha=0.7)
#             plt.plot(x, hilbert_torch[idx_slice, :], '--', label='PyTorch Hilbert2', alpha=0.7)
#             plt.title(f'2D Hilbert Transform - {desc} (Slice at y=0)')
#             plt.xlabel('x')
#             plt.ylabel('Hilbert Transform')
#             plt.legend()
#             plt.savefig(f'plots/plot_{plot_index}.png')
#             plt.close()
#             plot_index += 1

#             # Store Results
#             test_results.append({
#                 'Scenario': f'2D - {desc}',
#                 'Max Error': max_error,
#                 'Mean Error': mean_error
#             })

#     # Generate CSV Report
#     df_results = pd.DataFrame(test_results)
#     df_results.to_csv('hilbert_transform_test_results.csv', index=False)
#     print("Test results saved to 'hilbert_transform_test_results.csv'")

#     # Display the DataFrame
#     print(df_results)

# if __name__ == '__main__':
#     run_tests()


import torch
import numpy as np
import scipy.signal

def my_fftn(x, s=None, dim=None, norm='backward'):
    """
    Wrapper around torch.fft.fftn providing flexibility similar to numpy/scipy's fftn.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    s : tuple of ints, optional
        Shape along the transformed axes.
    dim : tuple of ints, optional
        Dimensions along which to take the FFT.
    norm : str, optional
        Normalization mode. Same as torch.fft.fftn norm parameter. ('backward', 'forward', 'ortho')
    
    Returns
    -------
    Xf : torch.Tensor
        The n-dimensional discrete Fourier Transform of x.
    """
    return torch.fft.fftn(x, s=s, dim=dim, norm=norm)

def my_ifftn(X, s=None, dim=None, norm='backward'):
    """
    Wrapper around torch.fft.ifftn providing flexibility similar to numpy/scipy's ifftn.
    
    Parameters
    ----------
    X : torch.Tensor
        Input tensor in frequency domain.
    s : tuple of ints, optional
        Shape along the transformed axes.
    dim : tuple of ints, optional
        Dimensions along which to take the iFFT.
    norm : str, optional
        Normalization mode. Same as torch.fft.ifftn norm parameter. ('backward', 'forward', 'ortho')
    
    Returns
    -------
    x : torch.Tensor
        The n-dimensional inverse discrete Fourier Transform of X.
    """
    return torch.fft.ifftn(X, s=s, dim=dim, norm=norm)

def _hilbert_filter_1d(N, device, dtype):
    """
    Construct a 1D Hilbert filter vector of length N.
    This matches the logic used in the original 1D hilbert function.
    
    Parameters
    ----------
    N : int
        Length of the axis being transformed.
    device : torch.device
        Device for the filter.
    dtype : torch.dtype
        Data type for the filter.
    
    Returns
    -------
    h : torch.Tensor
        1D Hilbert filter of shape [N]
    """
    h = torch.zeros(N, dtype=dtype, device=device)
    if N % 2 == 0:
        # even length
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        # odd length
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return h

def hilbert_nd(x, s=None, dim=None, norm='backward'):
    """
    Compute the N-dimensional analytic signal of x along specified dimensions using the Hilbert transform.
    
    Parameters
    ----------
    x : torch.Tensor
        Real input signal.
    s : tuple of ints, optional
        Shape along the transformed axes. Default: no zero-padding.
    dim : tuple of ints, optional
        Dimensions along which to compute the Hilbert transform. If None, all dimensions are used.
    norm : str, optional
        Normalization mode for FFT operations. ('backward', 'forward', 'ortho')
    
    Returns
    -------
    x_analytic : torch.Tensor
        Analytic signal of x along the specified dimensions.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if dim is None:
        dim = tuple(range(x.ndim))
    
    # Determine the sizes along transform dimensions
    orig_shape = x.shape
    if s is None:
        s = [orig_shape[d] for d in dim]

    # Compute FFT
    Xf = my_fftn(x, s=s, dim=dim, norm=norm)

    # Construct the N-D Hilbert filter as an outer product of 1D filters
    # for each dimension in dim.
    filters = []
    for i, d in enumerate(dim):
        N_d = s[i]
        h_d = _hilbert_filter_1d(N_d, Xf.device, Xf.dtype)
        filters.append(h_d)

    # Now we have a list of filters: h1 (shape [N_dim1]), h2 ([N_dim2])...
    # We must form the N-D filter by taking the outer product step by step.
    # Start with the first filter, then expand and multiply.
    H = filters[0]
    # For subsequent filters, do an outer product
    for i in range(1, len(filters)):
        # shape: we want to broadcast them, so we view them in a compatible way
        H = H.unsqueeze(-1) * filters[i].unsqueeze(0)
    # Now H is an N-D array of shape [N_dim1, N_dim2, ...]
    # We must match H to the full dimensionality of Xf:
    # Insert singleton dimensions for non-transformed axes at the end.
    # The order of dim might not be sorted, so let's handle carefully.
    # We'll permute H to match the order of dim relative to x.

    # Sort dim for easier handling
    dim_sorted = sorted(dim)
    # If dim is not sorted, we must reorder H accordingly.
    # H is constructed in the order of 'dim', so we must match this order.
    # If dim is not sorted, the shape of H matches the order of dim as given.
    # We'll permute the dimensions of x so that transform dims are at front.
    # Then we can match H easily.

    # Let's rearrange x to put transform dims in front (in order), apply H, then rearrange back.
    # A simpler approach: We'll directly expand H to match Xf shape:
    # We'll create a full shape with 1s in non-transform dims and put H in transform dims.
    full_shape = list(Xf.shape)
    # Replace the sizes of transform dims in full_shape with the shape of H
    for i, d in enumerate(dim):
        full_shape[d] = s[i]

    # Now we reshape H to full_shape. But we must place H's dimensions exactly in the transform dims.
    # The transform dims might not be consecutive or sorted. We must carefully place them:
    # Current H order is given by the order of 'dim', so we must insert 1s in places corresponding to non-dim axes.
    # Start by creating a list of ones for full_shape, then assign H:
    H_nd = torch.ones(full_shape, dtype=Xf.dtype, device=Xf.device)
    # We'll use advanced indexing to assign H along 'dim'.
    # Construct an indexing tuple:
    # For each axis of Xf, if it's in dim, we want all elements from H in that axis;
    # if it's not in dim, we want all from dimension 1 (just expand).
    # But direct assignment isn't trivial. Instead, let's carefully expand H.

    # Let's do this step by step:
    # H currently matches the order of dim exactly in creation.
    # We'll expand H so that it matches the full shape in the correct positions.
    # Initialize a list of dimension sizes for H_expanded:
    H_expanded_shape = [1]*Xf.ndim
    for i, d_ in enumerate(dim):
        H_expanded_shape[d_] = s[i]
    H_nd = H.view(*H_expanded_shape)

    # Multiply Xf by H_nd
    Xf = Xf * H_nd

    # Inverse FFT
    x_analytic = my_ifftn(Xf, s=s, dim=dim, norm=norm)

    return x_analytic


# ------------------- Testing the ND Hilbert Transform --------------------

if __name__ == "__main__":
    # We can test 1D and 2D cases against scipy to ensure correctness.

    # 1D Test
    t = np.linspace(0, 2 * np.pi, 500)
    x_np = np.sin(t)
    x_torch = torch.from_numpy(x_np)

    # PyTorch hilbert_nd (1D)
    x_analytic_torch = hilbert_nd(x_torch, dim=(-1,))
    hilbert_torch = x_analytic_torch.imag.numpy()

    # SciPy hilbert (1D)
    analytic_scipy = scipy.signal.hilbert(x_np)
    hilbert_scipy = np.imag(analytic_scipy)

    error_1D = np.max(np.abs(hilbert_torch - hilbert_scipy))
    print("1D Max Error:", error_1D)

    # 2D Test
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z_np = np.sin(X) + np.cos(Y)
    Z_torch = torch.from_numpy(Z_np)

    # PyTorch hilbert_nd (2D)
    # We'll transform along the first two dims (0,1)
    Z_analytic_torch = hilbert_nd(Z_torch, dim=(0,1))
    hilbert2_torch = Z_analytic_torch.imag.numpy()

    # SciPy hilbert2 (2D)
    analytic_scipy_2d = scipy.signal.hilbert2(Z_np)
    hilbert2_scipy = np.imag(analytic_scipy_2d)

    error_2D = np.max(np.abs(hilbert2_torch - hilbert2_scipy))
    print("2D Max Error:", error_2D)

    # Optional: Test higher dimensions (no direct SciPy comparison)
    # For demonstration, let's do a 3D random test:
    # We'll just ensure it runs and returns a complex tensor of the same shape.
    data_3d = torch.randn(20, 20, 20)
    data_3d_analytic = hilbert_nd(data_3d, dim=(0,1,2))
    print("3D hilbert transform shape:", data_3d_analytic.shape)
    print("3D hilbert transform dtype:", data_3d_analytic.dtype)
    print("3D hilbert transform device:", data_3d_analytic.device)
