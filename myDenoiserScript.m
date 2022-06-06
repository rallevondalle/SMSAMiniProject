%% Generate signal data and noise
dt     = .001;                              % time resolution, 'sample rate'
t      = 0:dt:1;                            % time vector
fclean = sin(2*pi*50*t) + sin(2*pi*120*t);  % sum of 2 frequencies
f      = fclean + 2.5*randn(size(t));       % add noise


% Plot signal and noise vectors
figure
subplot(3,1,1)
plot(t,f,'r','LineWidth',1.5), hold on
plot(t,fclean,'k','LineWidth',1.2)
l1 = legend('Noise','Data');set(l1,'FontSize',14)
ylim([-10 10]); set(gca,'FontSize',14)
xlabel('Time (s)')
ylabel('Amplitude')
hold off


% Compute FFT of noisy signal data
n       = length(t);
fhat    = fft(f,n);                     % compute the fast Fourier transform
PSD     = fhat.*conj(fhat)/n;           % calculate power spectrum
freq    = 1/(dt*n)*(0:n);               % create x-axis of frequencies in Hz
L       = 1:floor(n/2);                 % vector with only half of the frequencies


% Use PSD to filter noise
denoiseThreshold = 100;                 % denoise threshold
indices  = PSD>denoiseThreshold;        % select frequency components above threshold
PSDclean = PSD.*indices;                % zero out below threshold
fhat     = indices.*fhat;               % zero out small Fourier coeffs. in signal
ffilt    = ifft(fhat);                  % Inverse FFT


% Plot denoised data
subplot(3,1,2)
plot(t,ffilt,'blue')
legend('Filtered Data')
ylim([-10 10]); set(gca,'FontSize',14)
xlabel('Time (s)')
ylabel('Amplitude')


% Generate threshold vector for plot
noiseThreshold = ones(1,1/dt+1);
noiseThreshold = noiseThreshold*denoiseThreshold;
xVector        = [0:1000];


% Plot power spectrum of noise, filtered data, and threshold
subplot(3,1,3)
plot(freq(L),PSD(L),'r','LineWidth',1.5), hold on
plot(freq(L),PSDclean(L),'-b','LineWidth',1.2)
plot(xVector,noiseThreshold,'black')
xlim([0 500]);set(gca,'FontSize',14)
legend('Noise','Filtered Data','Threshold')
xlabel('Frequency (Hz)')
ylabel('Power')