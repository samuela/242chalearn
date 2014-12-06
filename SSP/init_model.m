function [ model, x0, P0 ] = init_model(num_states,D,M)
%    Initializes EM parameters for each of the 21 states.
%
%    Outputs:
%       model: a 1xnum_states cell-array, where model{k} corresponds to the initial
%              model for the k-th honeybee state. model{k} has four fields:
%              A,C,Q,R, which are defined in HW3.
%       x0: guess for the initial state for all models.
%       P0: covariance matrix on the initial state for all models.
%
% Brown CS242

    sig_theta = 0.05;

    model = cell([1,num_states]);
    for i=1:num_states
        model{i}.A = eye(D);
        model{i}.C = eye(M);
        model{i}.Q = eye(D);%blkdiag( eye(D/2), [ sig_theta, 0; 0, sig_theta ] );
        model{i}.R = eye(M);%blkdiag( eye(M/2), [ sig_theta, 0; 0, sig_theta ] );
    end
    x0 = zeros([D,1]);
    P0 = eye(D);

end
