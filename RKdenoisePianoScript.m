%% PIANO DENOISE TESTER AND RESYNTHESIZER
close all; clear all; clc;

[x, fs] = audioread('CTPiano8k.wav');   % load an audio file, fs = 8000
xmono = x(:, 1);                        % mono signal, select first channel

analysisLength = 3;                    % x seconds of audio

xshort = xmono(1:analysisLength*fs,1);  % select 1 second of audio
n = length(xshort);                     % number of samples to calculate
length = length(xshort);                % length of FFT
h = length/n;                           % sampling interval
t = (0:h:length-h);                     % time vector
S = xshort;                             % shorten sampled audio

% Random noise
RN = 0.25*randn(n,1);
NS = RN + S;                            % add noise to signal
NSmono = NS(1:n,1);                     % reduce dimensions to 1 channel

% Plots
figure(1)
subplot(3,1,1)
plot(t,RN,'k');
title('Generated Noise Data');set(gca,'FontSize',14)
ylabel('Amplitude')
ylim([-1 1])

subplot(3,1,2)
plot(t,S,'blue');
title('Sampled Clean Signal');set(gca,'FontSize',14)
ylabel('Amplitude')
ylim([-1 1])

subplot(3,1,3)
plot(t,NSmono);
title('Noisy Signal');set(gca,'FontSize',14)
xlabel('Time (s)')
ylabel('Amplitude')
ylim([-1 1])


% Compute FFT and PSD of noisy signal data
fhat    = fft(NSmono,n);                % compute the fast Fourier transform
PSD     = fhat.*conj(fhat)/n;           % calculate power spectrum
freq    = (0:n)/2;                      % create x-axis of frequencies in Hz
L       = 1:floor(n/2);                 % vector with only half of the frequencies

figure(2)
plot(freq(L),PSD(L),'r','LineWidth',1.5), hold on
title('Power Spectrum, Noisy Signal'); set(gca,'FontSize',14)


% Use PSD to filter noise
PSDthreshold = 1;                       % denoise threshold

indices      = PSD>PSDthreshold ;       % select frequency components above threshold
PSDclean     = PSD.*indices;            % zero out below threshold
indicesNoise = PSD<PSDthreshold ;       % select noise components, below PSD threshold
PSDnoise     = PSD.*indicesNoise;       % zero out above threshold
fhat         = indices.*fhat;           % zero out small Fourier coeffs. in signal
ffilt        = ifft(fhat);              % Inverse FFT


% Compute FFT and PSD of clean
fhatCleanS    = fft(S,n);               % compute the fast Fourier transform
PSDcleanS     = fhatCleanS.*conj(fhatCleanS)/n; % calculate power spectrum


% Uncomment to listen
%soundsc(NSmono,fs)
soundsc(ffilt,fs)

% Normalize audio to amplitudes between [-1 1]
NSmonoNorm = 0.9*(NSmono / max(NSmono));
ffiltNorm  = 0.9*(ffilt / max(ffilt));


% Export audio
%audiowrite('NSmono.wav',NSmonoNorm,fs);
%audiowrite('ffilt.wav',ffiltNorm,fs);
%audiowrite('original.wav',xshort,fs);


% Plot denoised data vs. original sample
figure(3)
subplot(2,1,1)
plot(t,ffilt,'m')
legend('Filtered Data')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 1500])
xlabel('Samples')
ylabel('Amplitude')
title('Denoised data vs. Original Sample')

subplot(2,1,2)
plot(t,S,'blue')
legend('Original Sample')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 1500])
xlabel('Samples')
ylabel('Amplitude')


% Generate threshold vector for plot
noiseThreshold = ones(1,n+1);
noiseThreshold = noiseThreshold*1;
xVector        = [0:n];


% Plot power spectrum of noise, filtered data, and threshold
figure(4)
subplot(2,1,1)
plot(freq(L),PSD(L),'r','LineWidth',1.5), hold on
plot(freq(L),PSDnoise(L),'-b','LineWidth',1.2)
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 2000]);set(gca,'FontSize',14)
legend('Filtered Data','Noise','Threshold')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum of Filtered (Denoised) Data & Noise Components')

subplot(2,1,2)
plot(freq(L),PSDcleanS(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 2000]);set(gca,'FontSize',14)
legend('Original Clean Sample')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum of Original Clean Sample')


% Overall MSE between sampled audio and filtered signal
MSE = norm(ffilt-S,'fro')^2/numel(S);


% Compute Mean Square Error for each sample
mse = zeros(1,n);                       % initialize mse vector

for i = 1:n                             % mse per sample
    mse(i) = mean((ffilt(i)-S(i)).^2);
end

figure(5)


% Plot Power Spectra of PSDs, noise, and threshold
subplot(2,1,1)
plot(freq(L),PSD(L),'r','LineWidth',1.5), hold on
plot(freq(L),PSDnoise(L),'-b','LineWidth',1.2)
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 2000]);set(gca,'FontSize',14)
legend('Filtered Data','Noise','Threshold')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum of denoised & reconstructed signal')

subplot(2,1,2)
plot(freq(L),PSDcleanS(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 2000]);set(gca,'FontSize',14)
legend('Original Sample')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum of original clean sample')


% Plot amplitude waveforms of sampled audio, resynthesized audio and MSE
figure(6)
subplot(3,1,1)
plot(t,ffilt,'m')
legend('Filtered Data')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 1500])
xlabel('Samples')
ylabel('Amplitude')
title('Denoised Data vs. Original Sample')

subplot(3,1,2)
plot(t,S,'blue')
legend('Original Sample')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 1500])
xlabel('Samples')
ylabel('Amplitude')

subplot(3,1,3)
area(abs(mse))
legend('Mean Square Error, MSE = .0063')
ylim([0 1]); set(gca,'FontSize',14)
xlim([0 1500])
xlabel('Samples')
ylabel('MSE')
ylim([0 1])
