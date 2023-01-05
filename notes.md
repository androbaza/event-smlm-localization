# Aims of the project
* Generate localizations from the event-based SMLM data using phasor based localization method.
* Adapt the input data to perform localization using DECODE.
* Generate 3D localizations?

# Workflow
* Filter out the event salt-n-paper noise --> not needed with doG
* There is no background - no median filtering?
* generating the images as floats reduces the number of false double peak identifications after doG filtereing. 

# TO-DO
* finish the script for pSMLM, make it a function in the end
* add custom parameters, so than grid search for parameters could be performed