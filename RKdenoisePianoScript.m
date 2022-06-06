%% PIANO DENOISE TESTER AND RESYNTHESIZER
close all; clear all; clc;

[x, fs] = audioread('CTPiano.wav');     % load audio file
xmono = x(:, 1);                        % mono signal, select first channel

analysisLength = 15;                    % x seconds of audio

xshort = xmono(1:analysisLength*fs,1);  % shorten audio for analysis
n = length(xshort);                     % number of samples to calculate
length = length(xshort);                % length of FFT
h = length/n;                           % sampling interval
t = (0:h:length-h);                     % time vector


% Apply fade in and fade out to original sample
ampFadeSamples = fs/100;
fadeInEnvelope = linspace(0, 1, ampFadeSamples)';
fadeOutEnvelope = flip(fadeInEnvelope(1:ampFadeSamples));
fadeOutStartxshort = numel(xshort)-ampFadeSamples;
xshort(1:ampFadeSamples) = xshort(1:ampFadeSamples) .* fadeInEnvelope(1:ampFadeSamples);
xshort(fadeOutStartxshort+1:end) = xshort(fadeOutStartxshort+1:end) .* fadeOutEnvelope(1:ampFadeSamples);


% Random noise
S = xshort;                             % shortened, mono, original sampled audio
RN = 0.25*randn(n,1);
NS = RN + S;                            % add noise to signal
NS = NS/max(NS);                        % normalize to values [-1 1]
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
PSDmax  = max(PSD);                     % maximum power of signal
freq    = (0:n)/2;                      % create x-axis of frequencies in Hz
L       = 1:floor(n/2);                 % vector with only half of the frequencies

figure(2)
subplot(2,1,1)
plot(freq(L),PSD(L),'r','LineWidth',1.5)
xlim([0 fs/4])
xlabel('Frequency')
ylabel('Power')
title('Power Spectrum, Noisy Signal'); set(gca,'FontSize',14)

subplot(2,1,2)
plot(freq(L),PSD(L),'r','LineWidth',1.5)
xlim([0 1200])                          % set xlim for desired window
ylim([0 PSDmax/3])                      % set ylim for desired window
xlabel('Frequency')
ylabel('Power')
title('Power Spectrum zoomed, Noisy Signal'); set(gca,'FontSize',14)


% Use PSD to filter noise
PSDthreshold = 1;                       % denoise threshold
HPThreshold  = 10;                      % High-pass threshold in Hz

indices      = PSD>PSDthreshold;        % select frequency components above threshold
indices(1:HPThreshold)= 0;              % High-pass frequencies < Threshold
PSDclean     = PSD.*indices;            % zero out below threshold
indicesNoise = PSD<PSDthreshold;        % select noise components, below PSD threshold
PSDnoise     = PSD.*indicesNoise;       % zero out above threshold
fhat         = indices.*fhat;           % zero out small Fourier coeffs. in signal
ffilt        = ifft(fhat);              % Inverse FFT


% Apply fade in to reconstructed signal
ampFadeSamples = fs/100;
fadeInEnvelope = linspace(0, 1, ampFadeSamples)';
fadeOutEnvelope = flip(fadeInEnvelope(1:ampFadeSamples));
fadeOutStartffilt = numel(ffilt)-ampFadeSamples;
ffilt(1:ampFadeSamples) = ffilt(1:ampFadeSamples) .* fadeInEnvelope(1:ampFadeSamples);
ffilt(fadeOutStartffilt+1:end) = ffilt(fadeOutStartffilt+1:end) .* fadeOutEnvelope(1:ampFadeSamples);


% Normalize ffilt to S (original sample)
ffiltNormalizedToS = ffilt * (max(S) / max(ffilt));
fhatNoiseComponents = indicesNoise.*fft(NSmono,n);  % zero out small Fourier coeffs. in signal
noiseComponents = ifft(fhatNoiseComponents);
noiseComponents = 0.9*(noiseComponents/max(noiseComponents)); % Normalize noise [-1 1]


% Compute FFT and PSD of original sample
fhatS    = fft(S,n);               % compute the fast Fourier transform
PSDS     = fhatS.*conj(fhatS)/n; % calculate power spectrum


% Uncomment to listen
%soundsc(NSmono,fs)
soundsc(ffiltNormalizedToS,fs)


% Normalize audio to amplitudes between [-1 1]
NSmonoNorm = 0.9*(NSmono / max(NSmono));
ffiltNorm  = 0.9*(ffilt / max(ffilt));


% Plot denoised data vs. original sample
figure(3)
subplot(3,1,1)
plot(t,ffiltNormalizedToS,'m')
legend('Denoised Data')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 fs/5])
xlabel('Samples')
ylabel('Amplitude')
title('Denoised Data (normlized to Original Sample)')

subplot(3,1,2)
plot(t,S,'blue')
legend('Original Sample')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 fs/5])
xlabel('Samples')
ylabel('Amplitude')
title('Original Sample')

subplot(3,1,3)
plot(t,S,'blue'), hold on
plot(t,ffiltNormalizedToS,'m')
legend('Original Sample','Denoised Data')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 fs/5])
xlabel('Samples')
ylabel('Amplitude')
title('Denoised Data vs. Original Sample')


% Generate threshold vector for plot
noiseThreshold = ones(1,n+1);
noiseThreshold = noiseThreshold*1;
xVector        = [0:n];


% Plot power spectrum of noise, denoised data, and threshold
figure(4)
subplot(2,1,1)
plot(freq(L),PSDclean(L),'r','LineWidth',1.5), hold on
plot(freq(L),PSDnoise(L),'-b','LineWidth',1.2)
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/10])
legend('Denoised Data','Noise','Threshold')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum of Denoised Data & Noise Components')

subplot(2,1,2)
plot(freq(L),PSDS(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/10])
legend('Original Sample')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum of Original Sample')


% Overall MSE between sampled audio and denoised signal
MSE = norm(ffilt-S,'fro')^2/numel(S);


% Compute Mean Square Error for each sample
mse = zeros(1,n);                       % initialize mse vector

for i = 1:n                             % mse per sample
    mse(i) = mean((ffilt(i)-S(i)).^2);
end


% Plot Power Spectra of PSDs, noise, and threshold
figure(5)
subplot(3,1,1)
plot(freq(L),PSDS(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/10])
legend('Original Sample')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectra; Original Sample')

subplot(3,1,2)
plot(freq(L),PSDclean(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/10])
legend('Denoised Signal')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Denoised Signal')

subplot(3,1,3)
plot(freq(L),PSDnoise(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/10])
legend('Noise Components')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Noise Components')


% Plot amplitude waveforms of sampled audio, resynthesized audio and MSE
figure(6)
subplot(3,1,1)
plot(t,ffiltNormalizedToS,'m')
legend('Denoised Data, normed to Original Signal')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 4000])
xlabel('Samples')
ylabel('Amplitude')
title('Denoised Data vs. Original Sample')

subplot(3,1,2)
plot(t,S,'blue')
legend('Original Sample')
ylim([-1 1]); set(gca,'FontSize',14)
xlim([0 4000])
xlabel('Samples')
ylabel('Amplitude')

subplot(3,1,3)
area(abs(mse))
legend(sprintf('Mean Square Error = %.4f',MSE))
ylim([0 1]); set(gca,'FontSize',14)
xlim([0 4000])
xlabel('Samples')
ylabel('MSE')
ylim([0 1])


% Arrange figures
RKautoArrangeFigures(2,3,1,6)


%% Print figures to PNG
mkdir(sprintf('images/%i',fs));

for i = 1:6
    fig = figure(i);
    fig = sprintf('images/%i/figure_%i_sec_%i',fs,analysisLength,i);
    print(fig,'-dpng')
end


%% Export audio
mkdir(sprintf('audioExports/%i',fs));

audiowrite(sprintf('audioExports/%i/%i_NoisySignalNormalized.wav',fs,analysisLength),NSmonoNorm,fs);
audiowrite(sprintf('audioExports/%i/%i_ffiltNormalized.wav',fs,analysisLength),ffiltNorm,fs);
audiowrite(sprintf('audioExports/%i/%i_originalAudioShort.wav',fs,analysisLength),xshort,fs);
audiowrite(sprintf('audioExports/%i/%i_noiseComponents.wav',fs,analysisLength),noiseComponents,fs);


%% Helper functions

% FIGURE ARRANGER
function RKautoArrangeFigures(NH, NW, monitor_id, totalWidth)
% INPUT  :
%        NH : number of grid of vertical direction
%        NW : number of grid of horizontal direction
%        totalWidth : ajust figures to left side of screen [0 10]
% OUTPUT :
%
% get every figures that are opened now and arrange them.
%
% autoArrangeFigures selects automatically Monitor1.
% If you are dual(or more than that) monitor user, I recommend to set wide
% monitor as Monitor1.
%
% if you want arrange automatically, type 'autoArrangeFigures(0,0)' or 'autoArrangeFigures()'. 
%    But maximum number of figures for automatic mode is 27.
%
% if you want specify grid for figures, give numbers for parameters.
%    but if your grid size is smaller than required one for accommodating
%    all figures, this function changes to automatic mode and if more
%    figures are opend than maximum number, then it gives error.
%
% Notes
%  + 2017.1.20 use monitor id(Adam Danz's idea)
%
% leejaejun, Koreatech, Korea Republic, 2014.12.13
% jaejun0201@gmail.com

if nargin < 2
    NH = 0;
    NW = 0;
    monitor_id = 1;
    totalWidth = 6;
end

task_bar_offset = [0 0];

%
N_FIG = NH * NW;
if N_FIG == 0
    autoArrange = 1;
else
    autoArrange = 0;
end
figHandle = sortFigureHandles(findobj('Type','figure'));
n_fig = size(figHandle,1);
if n_fig <= 0
    warning('figures are not found');
    return
end

screen_sz = get(0,'MonitorPositions');
screen_sz = screen_sz(monitor_id, :);

if totalWidth > 0
    scn_w = totalWidth*0.1*(screen_sz(3) - task_bar_offset(1));
else
scn_w = screen_sz(3) - task_bar_offset(1);
end
scn_h = screen_sz(4) - task_bar_offset(2);

scn_w_begin = screen_sz(1) + task_bar_offset(1);
scn_h_begin = screen_sz(2) + task_bar_offset(2);

if autoArrange==0
    if n_fig > N_FIG
        autoArrange = 1;
        warning('too many figures than you told. change to autoArrange');
    else
        nh = NH;
        nw = NW;
    end
end

if autoArrange == 1
    grid = [2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4;
            3 3 3 3 3 3 3 3 4 4 4 5 5 5 5 5 5 5 5 6 6 6 7 7 7 7 7]';
   
    if n_fig > length(grid)
        warning('too many figures(maximum = %d)',length(grid))
        return
    end
    
    if scn_w > scn_h
        nh = grid(n_fig,1);
        nw = grid(n_fig,2);
    else
        nh = grid(n_fig,2);
        nw = grid(n_fig,1);
    end
end


fig_width = scn_w/nw;
fig_height = scn_h/nh;

fig_cnt = 1;
for i=1:1:nh
    for k=1:1:nw
        if fig_cnt>n_fig
            return
        end
        fig_pos = [scn_w_begin+fig_width*(k-1) ...
            scn_h_begin+scn_h-fig_height*i ...
            fig_width ...
            fig_height];
        set(figHandle(fig_cnt),'OuterPosition',fig_pos);
        fig_cnt = fig_cnt + 1;
    end
end

end

function figSorted = sortFigureHandles(figs)
    [tmp, idx] = sort([figs.Number]);
    figSorted = figs(idx);
end