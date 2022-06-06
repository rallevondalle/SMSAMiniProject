close all; clear all; clc;

[X, f] = myGaussianGenerator(-15, 50, -40, 40, 1000);
[X2, f2] = myGaussianGenerator(15, 50, -40, 40, 1000);


figure
area(X,f)
hold on
area(X2,f2)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
legend('Temporal Gaussian', 'Spectral Gaussian')
title('Optimal balance for STFT / Spectrogram')
set(gca,'XTick',[], 'YTick', [])
ylim([0 0.068])
hold off

%print('images/STFTGaussiansExample','-dpng')