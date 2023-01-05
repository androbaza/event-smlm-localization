from helpers import *

filepath = '/home/smlm-workstation/event-smlm/Evb-SMLM/generated_frames/tubulin300x400_frames_scaled_x1000_5.0ms-absolute_time_5_15.0.tif'

G_s1 = 2.5  # Sigma of the 1st Gaussian Filter
G_s2 = 6  # Sigma of the 2nd Gaussian Filter
local_max_scale = 7 # amount w/ the std of difference of gaussians is scaled
roi_rad = 4

im = io.imread(filepath)
frame = im[150,:,:]

# perfom difference of gaussians
G1 = gaussian_filter(frame, sigma=G_s1).astype(float)
G2 = gaussian_filter(frame, sigma=G_s2).astype(float)
doG = G1-G2

# min value for finding local max
min_val_LM = np.std(doG) * 7
peaks = peak_local_max(doG, threshold_abs = min_val_LM)

# remove peaks where they overlap with image borders
peaks = torch.tensor([peak for peak in peaks if \
    not peak[0]+roi_rad >=frame.shape[1] \
    and not peak[0]-roi_rad <= 0 \
    and not peak[1]+roi_rad >=frame.shape[2] \
    and not peak[1]-roi_rad <= 0])

# create a list of tensors from areas around peaks
roi = torch.tensor([im[peak[0]-roi_rad:peak[0]+roi_rad, peak[1]-roi_rad:peak[1]+roi_rad] for peak in peaks])