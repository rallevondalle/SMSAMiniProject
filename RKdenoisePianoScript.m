%% PIANO DENOISE TESTER AND RESYNTHESIZER
close all; clear all; clc;

scenario = 3;

if scenario == 1
    filename = '170CSolitudePiano1SP.wav'; % 4.5 seconds analysis, PSDthreshold 0.6
    analysisSetup = [4.5 ; 0.6];
elseif scenario == 2
    filename = 'rkPianoF3OneNote.wav';     % 2 seconds analysis, PSDthreshold 0.5
    analysisSetup = [2.0 ; 0.5];
elseif scenario == 3
    filename = 'CTPiano.wav';              % 2 seconds analysis, PSDthreshold 0.5
    analysisSetup = [10.0 ; 1.0];           % analysisLength, PSDthreshold
end

%filename = 'RasSyngerTest.wav';
%filename = 'rkPianoF086BPM.wav';


[pathstr,name,ext] = fileparts(filename);
[x, fs] = audioread(filename);          % load audio file
xmono = x(:, 1);                        % mono signal, select first channel

analysisLength = analysisSetup(1);      % seconds of audio to analyse
PSDthreshold = analysisSetup(2);        % denoise threshold

HPThreshold  = 2;                       % High-pass threshold in Hz
noiseAmount = 0.25;                     % amplitude of noise signal to be added

maxAnalysisLength = floor(length(x)/fs);
if analysisLength < maxAnalysisLength
    analysisLength = analysisLength;
else
analysisLength = maxAnalysisLenght;
end

xshort = xmono(1:analysisLength*fs,1);  % shorten audio for analysis

%n = length(xshort);                    % number of samples to calculate
n = 2^nextpow2(length(xshort));         % next power of 2 for FFT efficiency
xshort(end+1:n) = 0;                    % zero pad x to next power of 2
sampleLength = length(xshort);          % length of FFT
t = (0:1:sampleLength-1);               % time vector


% Apply fade in and fade out to original sample
ampFadeSamples = fs/100;
fadeInEnvelope = linspace(0, 1, ampFadeSamples)';
fadeOutEnvelope = flip(fadeInEnvelope(1:ampFadeSamples));
fadeOutStartxshort = numel(xshort)-ampFadeSamples;
xshort(1:ampFadeSamples) = xshort(1:ampFadeSamples) .* fadeInEnvelope(1:ampFadeSamples);
xshort(fadeOutStartxshort+1:end) = xshort(fadeOutStartxshort+1:end) .* fadeOutEnvelope(1:ampFadeSamples);


% Random noise
S = xshort;                             % shortened, mono, original sampled audio
RN = noiseAmount*randn(n,1);
NS = RN + S;                            % add noise to signal
NS = NS/max(NS);                        % normalize to values [-1 1]
NSmono = NS(1:n,1);                     % reduce dimensions to 1 channel


% Plots
figure(1)
subplot(3,1,1)
plot(t,RN,'k');
title(sprintf('Generated Noise Data, %s',name));set(gca,'FontSize',14)
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
title('Denoised Data (normalized to Original Sample)')

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
noiseThreshold = ones(1,n+1) .* PSDthreshold;
noiseThreshold = noiseThreshold*1;
xVector        = 0:n;


% Plot power spectrum of noise, denoised data, and threshold
figure(4)
subplot(2,1,1)
plot(freq(L),PSDclean(L),'r','LineWidth',1.5), hold on
plot(freq(L),PSDnoise(L),'-b','LineWidth',1.2)
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/20])
legend('Denoised Data','Noise','Threshold')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectrum of Denoised Data & Noise Components')

subplot(2,1,2)
plot(freq(L),PSDS(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/20])
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
ylim([0 PSDmax/20])
legend('Original Sample')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Power Spectra; Original Sample')

subplot(3,1,2)
plot(freq(L),PSDclean(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/20])
legend('Denoised Signal')
xlabel('Frequency (Hz)')
ylabel('Power')
title('Denoised Signal')

subplot(3,1,3)
plot(freq(L),PSDnoise(L),'r','LineWidth',1.5), hold on
plot(xVector,noiseThreshold,'k','LineWidth',2,'LineStyle','--')
xlim([0 4000]);set(gca,'FontSize',14)
ylim([0 PSDmax/20])
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


% Spectrogram of Signal
% analysis parameters
wlen = 1024;                            % window length (power of 2 best)
hop = wlen/4;                           % hop size (power of 2 best)
nfft = 4096;                            % number of fft points (power of 2 best)

% perform STFT
win = blackman(wlen, 'periodic');       % window for FFT
[S1, ffft1, tfft1] = stft(xshort, win, hop, nfft, fs);

% calculate the coherent amplification of the window
C1 = sum(win)/wlen;

% take the amplitude of fft(x) and scale it, so not to be a
% function of the length of the window and its coherent amplification
S1 = abs(S1)/wlen/C1;

% correction of the DC & Nyquist component
if rem(nfft, 2)                     % odd nfft excludes Nyquist point
    S1(2:end, :)   = S1(2:end, :) .* 2;
else                                % even nfft includes Nyquist point
    S1(2:end-1, :) = S1(2:end-1, :) .* 2;
end

% convert amplitude spectrum to dB (min = -120 dB)
S1 = 20*log10(S1 + 1e-6);



% perform STFT
win = blackman(wlen, 'periodic');       % window for FFT
[S2, ffft2, tfft2] = stft(ffilt, win, hop, nfft, fs);

% calculate the coherent amplification of the window
C2 = sum(win)/wlen;

% take the amplitude of fft(x) and scale it, so not to be a
% function of the length of the window and its coherent amplification
S2 = abs(S2)/wlen/C2;

% correction of the DC & Nyquist component
if rem(nfft, 2)                     % odd nfft excludes Nyquist point
    S2(2:end, :)   = S2(2:end, :) .* 2;
else                                % even nfft includes Nyquist point
    S2(2:end-1, :) = S2(2:end-1, :) .* 2;
end

% convert amplitude spectrum to dB (min = -120 dB)
S2 = 20*log10(S2 + 1e-6);


% plot the spectrogram
figure(7)
subplot(2,1,1)
surf(tfft1, ffft1, S1)
shading interp
axis tight
view(0, 90)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Amplitude spectrogram of signal')

hcol = colorbar;
set(hcol, 'FontName', 'Times New Roman', 'FontSize', 14)
ylabel(hcol, 'Magnitude, dB')

subplot(2,1,2)
surf(tfft2, ffft2, S2)
shading interp
axis tight
view(0, 90)
set(gca,'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Amplitude spectrogram of signal')

hcol = colorbar;
set(hcol, 'FontName', 'Times New Roman', 'FontSize', 14)
ylabel(hcol, 'Magnitude, dB')



% Arrange figures
RKautoArrangeFigures(3,3,1,5)


%% Print figures to PNG
mkdir(sprintf('images/%i',fs));

for i = 1:7
    fig = figure(i);
    fig = sprintf('images/%i/%s_figure_%s_sec_%i',fs,name,int2str(analysisLength),i);
    print(fig,'-dpng')
end


%% Export audio
mkdir(sprintf('audioExports/%i',fs));

audiowrite(sprintf('audioExports/%i/%s_%s_NoisySignalNormalized.wav',fs,name,int2str(analysisLength)),NSmonoNorm,fs);
audiowrite(sprintf('audioExports/%i/%s_%s_ffiltNormalized.wav',fs,name,int2str(analysisLength)),ffiltNorm,fs);
audiowrite(sprintf('audioExports/%i/%s_%s_originalAudioShort.wav',fs,name,int2str(analysisLength)),xshort,fs);
audiowrite(sprintf('audioExports/%i/%s_%s_noiseComponents.wav',fs,name,int2str(analysisLength)),noiseComponents,fs);


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