%% Fourier series approximation of 1 cycle square wave function
clear all, close all, clc

% Setup
dx = 0.01;                          % sample rate
L = 10;                             % length of sample
x = 0:dx:L;                         % x vector
n = length(x);                      % sampling points

% Generate Square Wave, 1 cycle
nquart = floor(n/4);                
f = zeros(size(x));
f(nquart:3*nquart) = 1;

% Compute Fourier coefficients
A0 = sum(f.*ones(size(x)))*dx*2/L;
fFS = A0/2;

fModes = 500;
drawInterval = 0.01*fModes;
C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]}; % cell array of colros.


figure(1)
set(gcf,'Position',[600 500 1800 1200])
plot(x,f,'k','LineWidth',2), hold on
xlim([0 L])
ylim([-0.2 1.2])

pause(2)

for k=1:fModes
    Ak = sum(f.*cos(2*pi*k*x/L))*dx*2/L;
    Bk = sum(f.*sin(2*pi*k*x/L))*dx*2/L;
    fFS = fFS + Ak*cos(2*k*pi*x/L) + Bk*sin(2*k*pi*x/L);
    if rem(k,drawInterval) == 0               % draw every xth Fourier mode
        for ii=1:length(C)
            plot(x,f,'k','LineWidth',2), hold on
            plot(x,fFS,'color',C{1+rem(ii+randi(2),length(C))},'LineWidth',1.2)
            drawnow
            pause(0.03)
            ii = ii+1;
        end
    title(sprintf('Mode %i of %i', k,fModes)),set(gca,'FontSize',14)
    end
end

legend('f(x)','Fourier Approximation'), set(gca,'FontSize',14)
title(sprintf('%i Fourier modes drawn every %i modes', fModes,drawInterval))

figure(2)
plot(x,f,'k','LineWidth',2), hold on
plot(x,fFS,'r--','LineWidth',1.2)
legend('f(x)','Fourier Approximation'), set(gca,'FontSize',14)
title(sprintf('Final approximation of %i Fourier modes', fModes))

%%
for i=1:2
    fig = figure(i);
    fig = sprintf('images/fouriermodesNEW_%s_every_%s_modes_%i',int2str(fModes),int2str(drawInterval),i);
    print(fig,'-dpng')
end
%RKautoArrangeFigures(2,2,1,5)
