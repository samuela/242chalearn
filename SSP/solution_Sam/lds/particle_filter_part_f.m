function [X] = particle_filter_part_f(num_particles, model, data, x0, P0, corruption)
%function [X] = particle_filter(num_particles, model, data, x0, P0)
% Inputs:
%    num_particles: number of particles to use
%    model: the data model structure, containing the fields A,C,Q,R, as
%           defined in the handout
%    data: T x 1 vector containing the time-series data
%    x0: guess for the initial state
%    P0: covariance matrix on the initial state
% Outputs:
%    X: D x T matrix containing the D-dimensional posterior mean of the
%       estimate for the states from time t=1 to time=T
%
% Brown CS242

  D = numel(x0);
  T = numel(data);
  X = zeros(D, T);
  
  % P is D x num_particles
  P = mvnrnd(x0, P0, num_particles)';
  w = ones(num_particles, 1);
  for t=1:T
    % Re-weight particles
    w = w .* (corruption * normpdf(data(t), 0, 40^2) ...
        + (1 - corruption) * mvnpdf(data(t), (model.C * P)', model.R));
    w = w / sum(w);
    
    % Calculate posterior mean
    X(:,t) = P * w;
    
    % Resample
    ix = randsample(1:num_particles, num_particles, true, w);
    P = P(:,ix);
    w = ones(num_particles, 1) / num_particles;
    
    % Propogate via transition probability
    P = mvnrnd((model.A * P)', model.Q)';
  end
end
