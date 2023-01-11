import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
from skimage.feature.peak import peak_local_max
# import os
# import torch
# import torch.fft as fft
import matplotlib.pyplot as plt
import time
import datetime
# from numba import njit

# filepath = '/home/smlm-workstation/event-smlm/Evb-SMLM/generated_frames/tubulin300x400_frames_scaled_x1000_5.0ms-absolute_time_5_15.0.tif'
filepath = '/home/smlm-workstation/event-smlm/Evb-SMLM/generated_frames/tubulin300x400_5_15_frames_scaled_x1000_8.0ms-absolute_time_0_9.999999.tif'

def extract_roi(frame, G_s1=2.5, G_s2=4, local_max_scale=7, roi_rad=5):
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
    # @njit(cache=True)
    def remove_border_peaks(peaks):
        return [peak for peak in peaks if
                not peak[0]+roi_rad >=frame.shape[1] \
                and not peak[0]-roi_rad <= 0 \
                and not peak[1]+roi_rad >=frame.shape[0] \
                and not peak[1]-roi_rad <= 0]

    # @njit(cache=True)
    def gen_rois(peaks):
        return [frame[peak[0]-roi_rad:peak[0]+roi_rad+1, 
                      peak[1]-roi_rad:peak[1]+roi_rad+1] for peak in peaks]

    # perfom difference of gaussians
    doG = gaussian_filter(frame, sigma=G_s1) - gaussian_filter(frame, sigma=G_s2)

    # find local peaks in doG
    peaks = peak_local_max(doG, threshold_abs = np.std(doG) * local_max_scale)

    # remove peaks which overlap with image borders
    peaks = remove_border_peaks(peaks)

    # create a list of tensors from areas around peaks
    return gen_rois(peaks), peaks


def fit_phasor(roi, peaks, roi_rad=3):
    """
    Args:
        roi : array_like
            input ROIs
        peaks : array_like
    Returns: 
        List : [X, Y, intensity]
    """
    # @njit(cache=True, fastmath=True)
    def est_coord(roi_ft, coord_type):
        phase_angle = np.arctan(roi_ft[coord_type].imag / roi_ft[coord_type].real) - np.pi
        return np.abs(phase_angle) / (2*np.pi/(roi_rad*2+1))

    localizations = np.zeros((len(roi), 3))

    for id, single_roi in enumerate(roi):
        # FT the ROIs
        # roi_ft = fft.fft2(single_roi)
        roi_ft = np.fft.fft2(single_roi)
        # find phase angles
        localizations[id, :] = peaks[id][1] + est_coord(roi_ft, (0, 1)) - roi_rad, \
                               peaks[id][0] + est_coord(roi_ft, (1, 0)) - roi_rad, \
                               np.sum(np.nonzero(single_roi))
    return localizations
    
im = io.imread(filepath)
# frame = im[150,:,:]
# start = time.perf_counter()
# # localizations = fit_phasor(*extract_roi(frame))
# print(time.perf_counter() - start)

# im = im[:100,:,:]

start = time.perf_counter()
localizations_whole, num_loc = np.zeros((100000,4)), 0
for id, _ in enumerate(im):
    localizations = fit_phasor(*extract_roi(im[id,:,:]))
    localizations_whole[num_loc:num_loc+localizations.shape[0], 0] = id
    localizations_whole[num_loc:num_loc+localizations.shape[0], 1:4] = localizations
    num_loc += localizations.shape[0]
    if id%100==0: print(id)

localizations_whole = localizations_whole[:num_loc]
# np.savetxt('/home/smlm-workstation/event-smlm/event-smlm-localization/localizations/LocalizationList_'+ str(datetime.datetime.now())[:-7] +'.xml', localizations_whole,
#            fmt='%1.0f %1.4f %1.4f %1.1f', delimiter=",")
np.savetxt('/home/smlm-workstation/event-smlm/event-smlm-localization/localizations/LocalizationList_' + str(datetime.datetime.now())[:-7] + '.csv', localizations_whole,
           fmt='%1.0f %1.4f %1.4f %1.1f', delimiter=",", header='frame, x, y, Intensity')
print(time.perf_counter() - start)
print(localizations_whole.shape)

# plt.imshow(im[100,:,:], cmap='gray', vmin=0, vmax=3000)
# for l in range(0, 4000):
#   plt.scatter(localizations_whole[l, 1],
#               localizations_whole[l, 2], marker='x', s=5)


def merge_localizations(maxD=30, maxOff=5, maxL=500):
    pass
