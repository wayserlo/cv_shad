import numpy as np
from scipy import signal
from scipy.fft import fft2, fftshift, ifft2, ifftshift

def gaussian_kernel(size, sigma):
    #взято на семинаре, семинар просмотрен и осознан
    gkern1d = signal.windows.gaussian(size, std=sigma)
    gkern2d = gkern1d[None, :] * gkern1d[:, None]
    gkern2d /= gkern2d.sum()
    
    return gkern2d


def fourier_transform(kernel, shape):
    pad_h, pad_w = shape[0] - kernel.shape[0], shape[1] - kernel.shape[1]
    padding = [((pad_h+1) // 2, pad_h // 2), ((pad_w+1) // 2, pad_w // 2)]
    pad_kernel = np.pad(kernel, padding)
    kernel_shifted = np.roll(np.roll(fftshift(pad_kernel), shape[0] % 2, axis=0), shape[0] % 2, axis=1)
    kernel_fourier = fft2(kernel_shifted)
    
    return kernel_fourier


def inverse_kernel(H, threshold=1e-10):
    H_abs = np.abs(H)
    H[H_abs <= threshold] += 1
    H_inv = 1/H
    H_inv[H_abs <= threshold] = 0
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    shape = blurred_img.shape
    G = fourier_transform(blurred_img, shape) 
    H = fourier_transform(h, shape)
    H_inv = inverse_kernel(H, threshold=threshold)
    F_tilde = G * H_inv
    f_tilde = np.roll(np.roll(ifftshift(np.abs(ifft2(F_tilde))), -(shape[0] % 2), axis=0), -(shape[0] % 2), axis=1)
    
    return f_tilde


def wiener_filtering(blurred_img, h, K=0.00005):
    shape = blurred_img.shape
    G = fourier_transform(blurred_img, shape) 
    H = fourier_transform(h, shape)
    F_tilde = (np.conj(H) / (np.conj(H) * H + K)) * G
    f_tilde = np.roll(np.roll(ifftshift(np.abs(ifft2(F_tilde))), -(shape[0] % 2), axis=0), -(shape[0] % 2), axis=1)

    return f_tilde

def compute_psnr(img1, img2):
    mse = np.sum((np.float64(img1) - np.float64(img2)) ** 2) / np.prod(img1.shape)
    if mse == 0:
        raise ValueError("MSE is zero!")
    return 20 * np.log10( 255 / np.sqrt(mse))