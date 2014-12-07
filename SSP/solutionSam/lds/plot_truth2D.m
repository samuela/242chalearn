function plot_truth2D( data, figNum)

    useFigNum = 0;
    if(nargin == 2)
        useFigNum = 1;
    end
    if(useFigNum==1) figure(figNum); else figure; end;

    hold on;
    scatter( data(1,:), data(2,:), '.b ');
    xlabel('Position (x)');
    ylabel('Position (y)');
    hold off;
end

