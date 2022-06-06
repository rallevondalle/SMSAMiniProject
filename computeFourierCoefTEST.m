close all; clear all; clc;

[x, fs] = audioread('CTPiano.wav'); % load an audio file
x = x(:, 1);                        % get the first channel, mono
xshort = x(1:5*fs,1);

%downsample to 2000 Hz
x_trans = xshort';
[x2, fs2] = downsample(x_trans,441);

[freq,a,b] = computeFourierCoef(x,fs);
%plotFourierCoef(freq,a,b);

%%
a = -100; b = 100;
x = a + (b-a) * rand(1, 500);
mu = (a + b)/2;
sigma = 30; 
f = gaussian_distribution(x, mu, sigma);
plot(x,f,'.')
grid on
title('Gaussian Distribution Curve')
xlabel('X-axis')
ylabel('Gauss Distribution') 
function f = gaussian_distribution(x, mu, sigma)
p = -(1/2) * ((x - mu)/sigma) .^ 2;
A = 1/(sigma * sqrt(2*pi));
f = A.*exp(p); 
end