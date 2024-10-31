import numpy as np
from scipy.signal import convolve2d

def get_bayer_masks(n_rows, n_cols):
  rows = n_rows // 2 + n_rows % 2
  cols = n_cols // 2 + n_cols % 2
  red = np.array([[0, 1], [0,0]], dtype=bool)
  red = np.tile(red, (rows, cols))
  green = np.array([[1, 0], [0,1]], dtype=bool)
  green = np.tile(green, (rows, cols))
  blue = np.array([[0, 0], [1,0]], dtype=bool)
  blue = np.tile(blue, (rows, cols))
  if n_rows % 2 != 0:   #suboptimal way of working with odd matrices
    red = np.delete(red, -1, 0)
    green = np.delete(green, -1, 0)
    blue = np.delete(blue, -1, 0)
  if n_cols % 2 != 0:
    red = np.delete(red, -1, 1)
    green = np.delete(green, -1, 1)
    blue = np.delete(blue, -1, 1)
  return np.dstack((red, green, blue))


def get_colored_img(raw_img):
  mask = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
  raw_image = np.dstack((raw_img, raw_img, raw_img))
  return mask * raw_image



def bilinear_interpolation(colored_img):
  mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
  interpolated_image= colored_img.copy()
  kernel_rb = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])
  kernel_g = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])
  interpolated_image[..., 0] = np.uint8(convolve2d(colored_img[..., 0], kernel_rb, mode='same', boundary='fill', fillvalue=0))
  interpolated_image[..., 1] = np.uint8(convolve2d(colored_img[..., 1], kernel_g, mode='same', boundary='fill', fillvalue=0))
  interpolated_image[..., 2] = np.uint8(convolve2d(colored_img[..., 2], kernel_rb, mode='same', boundary='fill', fillvalue=0))
  return interpolated_image


def improved_interpolation(raw_img):
  alpha = 0.5
  beta = 0.625
  gamma = 0.75
  colored_img = get_colored_img(raw_img)
  kernel_g_rb = np.array([[0,0,-0.25,0,0], [0,0,0,0,0], [-0.25,0,1,0,-0.25], [0,0,0,0,0], [0,0,-0.25,0,0]])
  improved = np.float64(bilinear_interpolation(colored_img))
  #Блок с поправкой green color в R+B ячейках
  delta_gr = convolve2d(colored_img[..., 0], kernel_g_rb, mode='same', boundary='fill', fillvalue=0)
  delta_gb = convolve2d(colored_img[..., 2], kernel_g_rb, mode='same', boundary='fill', fillvalue=0)
  improved[..., 1] += alpha * (delta_gr + delta_gb) #с green закончили

  #Блок с корректировкой красного и голубого цветов, но в ячейках зеленого цвета
  kernel_rb_g_horizontal = np.array([[0,0,0.1,0,0], [0,-0.2,0,-0.2,0], [-0.2,0,1,0,-0.2], [0,-0.2,0,-0.2,0], [0,0,0.1,0,0]])
  kernel_rb_g_vertical = np.array([[0,0,-0.2,0,0], [0,-0.2,0,-0.2,0], [0.1,0,1,0,0.1], [0,-0.2,0,-0.2,0], [0,0,-0.2,0,0]])
  delta_rg_horizontal = convolve2d(colored_img[..., 1], kernel_rb_g_horizontal, mode='same', boundary='fill', fillvalue=0)
  delta_rg_horizontal[1::2] = 0
  delta_rg_vertical = convolve2d(colored_img[..., 1], kernel_rb_g_vertical, mode='same', boundary='fill', fillvalue=0)
  delta_rg_vertical[::2] = 0
  improved[..., 0] += beta * (delta_rg_horizontal + delta_rg_vertical) #поправка red в точках green


  delta_bg_horizontal = convolve2d(colored_img[..., 1], kernel_rb_g_horizontal, mode='same', boundary='fill', fillvalue=0)
  delta_bg_horizontal[::2] = 0
  delta_bg_vertical = convolve2d(colored_img[..., 1], kernel_rb_g_vertical, mode='same', boundary='fill', fillvalue=0)
  delta_bg_vertical[1::2] = 0
  improved[..., 2] += beta * (delta_bg_horizontal + delta_bg_vertical) #поправка blue в точках green

  #поправка RB в BR
  kernel_rb_br = np.array([[0,0,-0.25,0,0], [0,0,0,0,0], [-0.25,0,1,0,-0.25], [0,0,0,0,0], [0,0,-0.25,0,0]])
  delta_r_b = convolve2d(colored_img[..., 2], kernel_rb_br, mode='same', boundary='fill', fillvalue=0)
  improved[...,0] += gamma * delta_r_b #закончили с красным

  delta_b_r = convolve2d(colored_img[..., 0], kernel_rb_br, mode='same', boundary='fill', fillvalue=0)
  improved[...,2] += gamma * delta_b_r #закончили с синим
  return np.uint8(np.clip(improved, 0, 255))


def compute_psnr(img_pred, img_gt):
  mse = np.sum((np.float64(img_pred) - np.float64(img_gt)) ** 2) / np.prod(img_pred.shape)
  if mse == 0:
    raise ValueError("MSE is zero!")
  return 10 * np.log10((np.max(np.float64(img_gt)) ** 2 )/ mse)
