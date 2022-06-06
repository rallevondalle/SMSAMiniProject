function plotFourierCoef(f,a,b)
    % Plots the guitar coefficients
    % and adds labels

    % Plot the result
    figure("Position",[1,1,900,325])
    stem(f,a,"Filled")
    hold on
    stem(f,b,"Filled")
    hold off
    xlabel("Frequency (Hz)")
    ylabel("Coefficient")
    legend("a_n","b_n")
    title("Frequency domain")
    xlim([0,600])
end