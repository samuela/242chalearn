function [ model, x0, P0 ] = init_model_bee()
% function [ model, x0, P0 ] = init_model_bee()
%    Initializes EM parameters for each of the three honeybee states.
%
%    Outputs:
%       model: a 3x1 cell-array, where model{k} corresponds to the initial
%              model for the k-th honeybee state. model{k} has four fields:
%              A,C,Q,R, which are defined in the handout.
%       x0: guess for the initial state for all models.
%       P0: covariance matrix on the initial state for all models.
%
% Brown CS242

    D = 4;
    M = 4;
    num_states = 3;
    sig_theta = 0.05;

    for i=1:num_states
        model{i}.A = eye(D);
        model{i}.C = eye(M);
        model{i}.Q = blkdiag( eye(D/2), [ sig_theta, 0; 0, sig_theta ] );
        model{i}.R = blkdiag( eye(M/2), [ sig_theta, 0; 0, sig_theta ] );
    end
    x0 = zeros([D,1]);
    P0 = eye(D);

end
