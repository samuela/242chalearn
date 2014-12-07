function plot_truth(states, data, figNum)

    useFigNum = 0;
    if(nargin == 3)
        useFigNum = 1;
    end

    Nscans = size( states, 2 );
    if(useFigNum==1)
        figure(figNum);
    else
        figure;
    end;
    
    hold on;
    plot( 1:Nscans, states(1,:), '*r' );
    plot( 1:Nscans, [data], '.b' );
    xlabel('Time');
    ylabel('Position');
    hold off;
end

