close all; clear all; clc;

% number of signal measurements
n = 10000;
% measuring from 0 to 2 pi
length = 2*pi;
% difference between two measurements
h = length/n;

% steps
t = (0:h:length-h);
% Signal
S = sin(2*t)+cos(7*t)-cos(t);

% random noise
RN = 0.5*randn(n,1);

% adding the random noise to the signal
NS = transpose(RN) + S;

% getting the complex fourier coefficients using fft
ck = fft(NS);

% dividing the complex fourier coefficients by n 
ck = ck/n;

% setting any complex fourier coefficients smaller than 0.9 times the
% max to zero to remove the noise
m = max(ck);
for i = 1:n
  if ck(i) < 0.5*m
    ck(i) = 0;
  end
end


% getting the fourier coefficients ak and bk
s = floor(n/2)+1;
for i = 1:s
  ak(i) = 2 * real(ck(i));
  bk(i) = -2 * imag(ck(i));
end

% applying the fourier series in sin cos form
for i = 1:n
  y(i) = ak(1)/2;
  for j = 2:s
    y(i) = y(i) + ak(j) * cos((j-1)*i*h) + bk(j) * sin((j-1)*i*h);
  end
end

% plotting the results
plot(t,NS,'color','r'); hold on;
plot(t,y,'color','b'); hold on;
plot(t,S,'color','y'); hold on;
legend('noisy signal','generated signal', 'analytical signal')

return