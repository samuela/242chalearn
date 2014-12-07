function [res] = vectorize_seq(seq)
    % Takes a single honeybee sequence data and returns a vectorized
    % representation of the pose at each time step. In particular,
    % Inputs:
    %    seq: A single honeybee sequence data structure
    %
    % Outputs:
    %    res: Vectorized version of the honeybee pose. res(1:4,t)=
    %    represents the [x,y,sin,cos]' pose at time t.

    T = numel(seq.x);
    res = zeros(4,T);
    for(t=1:T)
        res(1,t) = seq.x(t);
        res(2,t) = seq.y(t);
        res(3,t) = seq.sin(t);
        res(4,t) = seq.cos(t);
    end
end

