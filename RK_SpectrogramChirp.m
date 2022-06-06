clear all, close all, clc

%%Quadratic Chirp Amplitude Waveform, Spectrogram, and Power Spectrum

fs = 5000;      % sample rate
dt = 1/fs;      % frequency resolution
t  = 0:dt:2;    % time vector
f0 = 50;        % start frequency of chirp
f1 = 1000;      % max frequency of chirp
t1 = 2;         % chirp length in seconds

x = chirp(t,f0,t1,f1,'quadratic');

soundsc(x,fs)
figure
plot(t,x)
t_plot = plot(t,x);
set(gca,'LineWidth',1.2,'FontSize',14);
title('Amplitude Waveform Quadratic Chirp')
xlabel('Time (s)')
ylabel('Amplitude')
print('images/AmplitudeWaveformQuadraticChirp','-dpng')

%%
figure
spectrogram(x,128,120,128,fs,'yaxis')   %hamming window, 128/2 + 1 = 65 frequency bins, 120 sample overlap
title('Spectrogram Quadratic Chirp')
colormap jet
set(gca,'LineWidth',1.2,'FontSize',14);
set(gcf,'Position',[100 100 550 200]);
print('images/SpectrogramQuadraticChirp','-dpng')

%%
n = length(t);
xhat = fft(x,n);
PDS = xhat.*conj(xhat)/n;
freq = 1/(dt*n)*(0:n);
L = 1:floor(n/2);

figure
plot(freq(L),PDS(L),'LineWidth',2.5)
set(gca,'FontSize',14)
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum Quadratic Chirp')
print('images/PowerSpectrumQuadraticChirp','-dpng')
