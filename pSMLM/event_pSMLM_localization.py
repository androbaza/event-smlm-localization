import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
from skimage.feature.peak import peak_local_max
import os
import torch
import torch.fft as fft

filepath = '/home/smlm-workstation/event-smlm/Evb-SMLM/generated_frames/tubulin300x400_frames_scaled_x1000_5.0ms-absolute_time_5_15.0.tif'


def extract_roi(frame, G_s1=2.5, G_s2=6, local_max_scale=7, roi_rad=5):
    """
    Args:
        frame (ndarray): input frame
        G_s1 (float): Sigma of the 1st Gaussian Filter.
        G_s1 (float): Sigma of the 2nd Gaussian Filter
        local_max_scale (int): coefficient w/w the std of difference of gaussians is scaled
        roi_rad (int): radius of ROI, best 5 or 7 at low photon counts
    Returns: 
        Tensor: list of ROIs
    """

    # perfom difference of gaussians
    doG = gaussian_filter(frame, sigma=G_s1) - gaussian_filter(frame, sigma=G_s2)

    # find local peaks in doG
    peaks = peak_local_max(doG, threshold_abs = np.std(doG) * local_max_scale)

    # remove peaks which overlap with image borders
    peaks = [peak for peak in peaks if \
        not peak[0]+roi_rad >=frame.shape[1] \
        and not peak[0]-roi_rad <= 0 \
        and not peak[1]+roi_rad >=frame.shape[2] \
        and not peak[1]-roi_rad <= 0]

    # create a list of tensors from areas around peaks
    return torch.tensor([im[peak[0]-roi_rad:peak[0]+roi_rad, peak[1]-roi_rad:peak[1]+roi_rad] for peak in peaks])

def fit_phasor(roi, roi_rad):
    """
    Args:
        roi (tensor): input ROIs
    Returns: 
        Tensor: localization coordinates, intensity
    """
    def roi_rad(roi_ft, coord_type):
        phase_angle = torch.atan(roi_ft[0,1].imag/roi_ft[0,1].real) - np.pi
        return abs(phase_angle)/(2*np.pi/(roi_rad*2+1))

    localizations = []

    for single_roi in roi:
        # FT the ROIs
        roi_ft = fft.fft2(single_roi)

        # find phase angles
        localizations.append()
    
im = io.imread(filepath)
frame = im[150,:,:]


