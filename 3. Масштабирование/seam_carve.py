import numpy as np

def compute_energy(image):
    Y_ext = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    I_part = np.zeros(np.prod(Y_ext.shape) * 2).reshape(Y_ext.shape[0], Y_ext.shape[1], 2)
    I_part[:, 1:-1, 0], I_part[:, 0, 0], I_part[:, -1, 0] = (Y_ext[:,2:]-Y_ext[:, :-2]) * 0.5, (Y_ext[:,1]-Y_ext[:,0]), (Y_ext[:,-1]-Y_ext[:,-2])
    I_part[1:-1, :, 1], I_part[0, :, 1], I_part[-1, :, 1] = (Y_ext[2:, :]-Y_ext[:-2, :]) * 0.5, (Y_ext[1, :]-Y_ext[0, :]), (Y_ext[-1, :]-Y_ext[-2, :])
    I_grad = np.sqrt(I_part[:,:,0] ** 2 + I_part[:,:,1] ** 2)
    return I_grad

def compute_seam_matrix(energy, mode, mask=None):
    if mask is not None:
        energy += mask * np.prod(mask.shape)*256
    seam_matrix = np.zeros(np.prod(energy.shape)).reshape(energy.shape[0], energy.shape[1])
    if mode == 'horizontal':
        seam_matrix[0,:] = energy[0,:]
        for i in range(1, seam_matrix.shape[0]):
            seam_matrix[i, 0] = min(seam_matrix[i-1,0], seam_matrix[i-1,1]) + energy[i, 0]
            seam_matrix[i, -1] = min(seam_matrix[i-1,-1], seam_matrix[i-1,-2]) + energy[i, -1]
            seam_matrix[i, 1:-1] = [min(seam_matrix[i-1,j-1], seam_matrix[i-1,j], seam_matrix[i-1,j+1]) + energy[i, j] for j in range(1, seam_matrix.shape[1]-1)]
    elif mode == 'vertical':
        seam_matrix = compute_seam_matrix(energy.T, 'horizontal').T
    return seam_matrix


def remove_minimal_seam(image, seam_matrix, mode, mask_img=None):
    image = image.astype('uint8')
    mask = np.zeros(image.shape[0]*image.shape[1], dtype='uint8').reshape(image.shape[0], image.shape[1])
    if mode == 'horizontal shrink':
        arg = np.argmin(seam_matrix[-1, :])
        mask[-1, arg] = 1
        for i in range (image.shape[0]-2, -1, -1):
            if arg == 0:
                arg += (np.argmin((seam_matrix[i, arg], seam_matrix[i, arg +1])))
            elif arg == image.shape[1]-1:
                arg += (np.argmin((seam_matrix[i, arg -1], seam_matrix[i, arg]))-1)
            else:
                arg += (np.argmin((seam_matrix[i, arg -1], seam_matrix[i, arg], seam_matrix[i, arg +1]))-1) 
            mask[i, arg ] = 1
        changed_img = image[np.stack((mask,mask,mask),axis=-1)==0].reshape(image.shape[0], image.shape[1]-1, image.shape[2]).copy()
        if mask_img is not None:
            changed_mask_img = mask_img[mask==0].reshape(mask_img.shape[0], mask_img.shape[1]-1).copy()
    elif mode == 'vertical shrink':
        if mask_img is not None:
            changed_img_T, changed_mask_imgT, mask_T = remove_minimal_seam(np.transpose(image, axes=(1, 0, 2)), seam_matrix.T, 'horizontal shrink',  mask_img.T)
            changed_img = np.transpose(changed_img_T, axes=(1, 0, 2))
            mask = mask_T.T
            changed_mask_img = changed_mask_imgT.T
        else:
            changed_img_T, _, mask_T = remove_minimal_seam(np.transpose(image, axes=(1, 0, 2)), seam_matrix.T, 'horizontal shrink',  mask_img)
            changed_img = np.transpose(changed_img_T, axes=(1, 0, 2))
            mask = mask_T.T
    if mask_img is not None:
        return (changed_img, changed_mask_img, mask)
    else:
        return (changed_img, None, mask)

def seam_carve(image, mode, mask=None):
    energy = compute_energy(image)
    seam_matrix = compute_seam_matrix(energy, mode.split()[0], mask)
    return remove_minimal_seam(image, seam_matrix, mode, mask)