# Aims of the project
* Generate localizations from the event-based SMLM data using phasor based localization method.
* Adapt the input data to perform localization using DECODE.
* Generate 3D localizations?

# Notes
* DoG is optimal for small spot sized below 5 pixels
* LoG is optimal for spot of size 5 - 20 pixels

# Workflow
* Filter out the event salt-n-paper noise --> not needed with doG
* There is no background - no median filtering?
* generating the images as floats reduces the number of false double peak identifications after doG filtereing. 
* for some reason numba performs worse for single image. Have to experiment with the whole movie ❌

# TO-DO
* finish the script for pSMLM, make it a function in the end ✅
* figure out how to properly calculate intensity of fluorophore.
* paralellization on CPU
* paralellization on GPU?
* add custom parameters, so than grid search for parameters could be performed
* DAOSTORM won't compile on Ubuntu 22
* check how PeakFit and Rain-STORM work