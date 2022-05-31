%% Computing the DFT matrix
clear all, close all, clc

n = 1024;               % sampling points
w = exp(-i*2*pi/n);

[I,J] = meshgrid(1:n,1:n);
DFT = w.^((I-1).*(J-1));

imagesc(real(DFT))      % displays DFT matrix as an image that uses
                        % the full range of colors in the colormap