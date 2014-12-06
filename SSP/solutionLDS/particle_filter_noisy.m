function [X] = particle_filter_noisy(num_particles, model, data, x0, P0)
%function [X] = particle_filter_noisy(num_particles, model, data, x0, P0)

% Inputs:
%    num_particles: number of particles to use
%    model: the data model structure, containing the fields A,C,Q,R, as
%           defined in the handout
%    data: T x 1 vector containing the time-series data
%    x0: initial guess for the state
%    P0: covariance matrix on the initial state
% Outputs:
%    X: D x T matrix containing the D-dimensional posterior mean of the
%       estimate for the states from time t=1 to time=T

  D = numel(x0);
  T = numel(data);
  resample_interval = 5;
  
  % init stuff
  X = zeros(D, T);
  P = zeros([D, D, T]);
  W = zeros([num_particles, T]);
  x = zeros([num_particles, D, T]);
    
  % unpack alphabet
  [ A, C, Q, R ] = deal( model.A, model.C, model.Q, model.R );

  % sample initial particles
  x_sim = mvnrnd( repmat( x0', [num_particles,1]), P0 );
  log_w_sim = log( 0.9 * mvnpdf( repmat( data(1)', [num_particles,1] ), x_sim*C', R ) ...
    + 0.1 *  mvnpdf( repmat( data(1)', [num_particles,1] ), x_sim*C', R + 40.^2 ));

  % time recursion
  for t=2:T
      
    % normalize weights
    log_w_sim = log_w_sim - max( log_w_sim );
    W(:,t-1) = exp(log_w_sim) ./ sum(exp(log_w_sim));
        
    % resample particles?
    if mod(t,resample_interval)==0      
      cdf = cumsum( exp(log_w_sim) ) ./ sum( exp( log_w_sim ) );
      [~,I] = histc(rand(num_particles,1), [0;cdf]);
      x(:,:,t-1) = x_sim( I, : );
      W(:,t-1) = 1 / numel(W(:,t-1));
    else
      x(:,:,t-1) = x_sim;      
    end
    
    % compute moments
    X(:,t-1) = x(:,:,t-1)' * W(:,t-1);
    res = x(:,:,t-1) - repmat(X(:,t-1)',[num_particles,1]);
    Psum = 0;
    for i=1:num_particles
      Psum = Psum + W(i,t-1) * res(i,:)' * res(i,:);
    end
    P(:,:,t-1) = Psum;
    
    % propogate
    mu = x(:,:,t-1) * A';    
    x_sim = mvnrnd( mu, Q );
    lq = log( mvnpdf( x_sim, mu, Q ) );
    
    % compute weights    
    llhood = log( 0.9 * mvnpdf( data(t), x_sim * C', R ) + ...
      0.1 * mvnpdf( data(t), x_sim * C', R + 40.^2 ));    
    lprior = log( mvnpdf( x_sim, mu, Q ) );
    log_w_sim = log( W(:,t-1) ) + llhood + lprior - lq;
        
  end
  
  % compute final moments  
  T = T;
  x(:,:,T) = x_sim;
  log_w_sim = log_w_sim - max( log_w_sim );
  W(:,T) = exp(log_w_sim) ./ sum(exp(log_w_sim));
  X(:,T) = x(:,:,T)' * W(:,T);
  res = x(:,:,T) - repmat(X(:,T)',[num_particles,1]);
  Psum = 0;
  for i=1:num_particles
    Psum = Psum + W(i,T) * res(i,:)' * res(i,:);
  end
  P(:,:,T) = Psum;
  
end