%% Script to verify an idea on the generation of doughnuts shaped signals
% when doing SMLM with an event camera (Dynamic Vision Sensor)
% Leonardo Redaelli

% Idea: such sensors has a LOGARITHMIC response time in the MICROS, the
% OUTPUT is just a DELTA after a threshold has been surpassed. The INPUT is
% a step convolved with a reversed decaying exponential, so an exponential
% transient starting from 0 toward a given value.

% In this script I simulate a gaussian and suppose that each pixel has an
% exponential transient up to its true value. Each time a threshold is being
% surpassed, a signal containing the time at which it has been surpassed is
% returned. Moreover, the current voltage value gets stored and becomes the
% new value with which the detector make comparisons. The storage of that
% voltage value to make comparisons is approximated to be instantaneous.
% Therefore the threshold values are known and can already be preallocated.

% CONCLUSIONS: no doughnut shape signal is generated. That phenomena is an
% unexplained artifact of the detector. From this simulation we understood
% that a reconstructed frame of an event-camera can be obtained by applying
% the underneath transformation to an emCCD-camera frame, since the
% simulation of the first one is comparable to experimental results.

% functions
gaussian = @(x, var) exp(- x.^2 ./ (2*var));
activation_time = @(pixel_val, threshold, Tdet) -Tdet*log(1 - threshold ./ pixel_val);
count_activations_dT = @(pixel, T0, dT) sum(T0<pixel & pixel<=(T0+dT));

% data
sigma = 3;
vec = -20:20;
Tdet = 100e-6; % response time of the detector
dT = 200e-6; % equivalent integration time to build the output matrix
N_ph = 10000;
B = 500; % starting value of the threshold
N = 20; % 20 different threshold values
Iref = B; % Iref contains all the thresholds with which the detector 
% response signal will be compared (I assumed that once the threshold has
% been surpassed and the detector gave a delta output the new threshold is
% taken as the signal value when the threshold was surpassed)
% for a log10(x) to be > 0.3 means x > 2 => every time multiplied by 2
for i=1:(N-1)
    Iref = [Iref, B*2^i];
end

g = gaussian(vec, sigma);

input_img = N_ph .* g .* g';

% plot the input
figure(1)
imagesc(input_img)
colorbar

figure(2)
[X, Y] = meshgrid(vec,vec');
mesh(X, Y, input_img)
colorbar

output = zeros(size(input_img,1), size(input_img,2), N); % will contain all
% the times of activation of the pixels when an exponential transitory is
% considered

output_img = zeros(size(input_img,1), size(input_img,2), 2); % will contain 
% only the number of activations per each pixel in a given time dT

% build the output matrix, containing in each pixel along the 3rd dimension
% the activation time. For log(-smth) the pixel will contains an imaginary
% number. This values will have to be placed to 0
for i=1:size(input_img, 1)
    for j=1:size(input_img, 2)
        output(i,j,:) = activation_time(input_img(i,j), Iref, Tdet);
        for k=1:N
            output(imag(output(i,j,k))~=0) = 0;
        end
    end
end


% create the frames of an equivalent integration for the first dT time 
% after excitation and the second dT time
for i=1:size(input_img, 1)
    for j=1:size(input_img, 2)
        for k=0:1
            output_img(i,j,k+1) = count_activations_dT(output(i,j,:), dT*k, dT*(k+1));
        end
    end
end

% plot the outputs
% figure(3)
% imagesc(output_img(:,:,1))

figure(3)
subplot(2,1,1)
imagesc(output_img(:,:,1))
colorbar
subplot(2,1,2)
imagesc(output_img(:,:,2))
colorbar

figure(4)
[X, Y] = meshgrid(vec,vec');
mesh(X, Y, output_img(:,:,1))
colorbar
