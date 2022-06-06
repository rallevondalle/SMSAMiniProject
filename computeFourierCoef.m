function [freq,a,b] = computeFourierCoef(y,fs, modes)
    % Estimates the coefficients for the Fourier
    % series using the trapezoid rule

    Nmodes = 1000;
    yc = y(1e4:2e4);
    
    tc = 1/fs*(0:(length(yc)-1));
    L = tc(end)/2;                      % The domain is assumed to be 2L
    freq = (1:Nmodes)/(2*L);            % Frequencies

    % Estimate coefficients using trapezoid rule
    a = zeros(1,Nmodes);
    b = zeros(1,Nmodes);
    
    a0 = 1/L*trapz(tc,yc);
    
    for n = 1:Nmodes
        a(n) = 1/L*trapz(tc,cos(n*pi/L*tc).*yc);
        b(n) = 1/L*trapz(tc,sin(n*pi/L*tc).*yc);
    end
end