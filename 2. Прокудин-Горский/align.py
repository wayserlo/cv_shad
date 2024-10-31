import numpy as np

def get_parametres(img):
  height = img.shape[0] // 3 # разделение на 3
  delta_width = int(img.shape[1] * 0.1) #для удаления рамок
  delta_height = int(height * 0.1)
  return height, delta_width, delta_height


#порядок B-G-R
def get_channels(img):
  height, delta_width, delta_height = get_parametres(img)
  blue_ch = img[delta_height : height - delta_height, delta_width: -delta_width]
  green_ch = img[height + delta_height : 2 * height - delta_height, delta_width: -delta_width]
  red_ch = img[2 * height + delta_height : 3 * height - delta_height, delta_width: -delta_width]
  return np.dstack((blue_ch, green_ch, red_ch))


def fourier_matching(img1, img2):
  c = np.fft.ifft2(np.fft.fft2(img1) * np.conj(np.fft.fft2(img2)))
  argmax = c.argmax()
  pre_par = np.unravel_index(argmax, c.shape)
  par = np.array([pre_par, pre_par - np.array([c.shape[0], c.shape[1]])])
  return par.T[np.absolute(par.T) < c.shape[0] * 0.1]


def roll_img(img, blue, red):
  rolled = img.copy()
  rolled[:,:, 0] = np.roll(rolled[:,:, 0], blue[0], axis = 0)
  rolled[:,:, 0] = np.roll(rolled[:,:, 0], blue[1], axis = 1)

  rolled[:,:, 2] = np.roll(rolled[:,:, 2], red[0], axis = 0)
  rolled[:,:, 2] = np.roll(rolled[:,:, 2], red[1], axis = 1)
  return rolled[..., ::-1]



def align(img, green):
  img_ch = get_channels(img)
  y_b, x_b = fourier_matching(img_ch[..., 1], img_ch[..., 0])
  y_r, x_r = fourier_matching(img_ch[..., 1], img_ch[..., 2])
  #print('y_b, x_b = ', (y_b, x_b))
  #print('y_r, x_r = ', (y_r, x_r))

  height, delta_width, delta_height = get_parametres(img)

  b_row = green[0] - height - y_b
  b_col = green[1] - x_b
  r_row = green[0] + height - y_r
  r_col = green[1] - x_r
  return roll_img(img_ch, (y_b, x_b), (y_r, x_r)), (b_row, b_col), (r_row, r_col)
