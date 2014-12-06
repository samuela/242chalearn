function [X,s,W] = particle_filter_slds(num_particles, model, data, x0, P0)
% PARTICLE_FILTER - Particle filter for the switching state-space model.
%
% Brown CS242

  D = numel(x0);
  resample_interval = 1;
  num_states = numel( model{1}.T0 );
  
  % init stuff
  T = size(data,2);
  X = zeros(D, T);
  W = zeros([num_particles, T]);
  x = zeros([num_particles, D, T]);
  s = zeros([num_particles, T]);
  
  % unpack alphabet
  for k=1:num_states
    [ A{k}, C{k}, Q{k}, R{k} ] = ...
      deal( model{k}.A, model{k}.C, model{k}.Q, model{k}.R );
  end
    
  % sample initial particles
  T0_cdf = cumsum(model{1}.T0);
  [~,s_sim] = histc(rand(num_particles,1), [0;T0_cdf]);
  x_sim = mvnrnd( repmat( x0', [num_particles,1]), P0 );
  log_w_sim = zeros([num_particles,1]);
  for k=1:num_states
    I = find( s_sim == k );
    like_tgt = mvnpdf( repmat( data(:,1)', [numel(I),1] ), x_sim(I,:)*C{k}', R{k} + 0.1*eye(D) );
    like_noise = mvnpdf( repmat( data(:,1)', [numel(I),1] ), zeros([numel(I), D]), 5*eye(D) );
    log_w_sim(I) = log( 0.9*like_tgt + 0.1*like_noise );
  end

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
      s(:,t-1) = s_sim( I, : );
      W(:,t-1) = 1 / numel(W(:,t-1));
    else
      x(:,:,t-1) = x_sim;      
      s(:,t-1) = s_sim;
    end
    
    % compute moments    
    p_k = zeros([num_states,1]);
    for state=1:num_states
      I_state = find( s(:,t-1) == state );
      p_k(state) = sum( W(I_state,t-1) );
    end
    X(:,t-1) = x(:,:,t-1)' * W(:,t-1);
    res = x(:,:,t-1) - repmat(X(:,t-1)',[num_particles,1]);
    Psum = 0;
    for i=1:num_particles
      Psum = Psum + W(i,t-1) * res(i,:)' * res(i,:);
    end
    
    % propogate        
    lq = zeros([num_particles,1]);
    llhood = zeros([num_particles,1]);
    lprior = zeros([num_particles,1]);
    for k=1:num_states
      I = find( s(:,t-1) == k );
      if isempty(I), continue; end;
      
      % sample discrete state      
      cdf = cumsum(model{k}.T(k,:))';
      [~,s_sim(I)] = histc(rand(numel(I),1), [0;cdf]);
      
      % sample continuous state
      mu = x(I,:,t-1) * A{k}';    
      x_sim(I,:) = mvnrnd( mu, Q{k} );
      lq(I) = log( mvnpdf( x_sim(I,:), mu, Q{k} ) );
        
      % compute weights    
      like_tgt = mvnpdf( repmat( data(:,t)', [numel(I),1] ), x_sim(I,:)*C{k}', R{k} + 0.1*eye(D) );
      like_noise = mvnpdf( repmat( data(:,t)', [numel(I),1] ), zeros([numel(I), D]), 5*eye(D) );
      llhood(I) = log( 0.9*like_tgt + 0.1*like_noise );       
      lprior(I) = log( mvnpdf( x_sim(I,:), mu, Q{k} ) );
    
    end
    log_w_sim = log( W(:,t-1) ) + llhood + lprior - lq;
        
  end
  
  % compute final moments  
  x(:,:,T) = x_sim;
  log_w_sim = log_w_sim - max( log_w_sim );
  W(:,T) = exp(log_w_sim) ./ sum(exp(log_w_sim));  
  X(:,T) = x(:,:,T)' * W(:,T);
  res = x(:,:,T) - repmat(X(:,T)',[num_particles,1]);
  Psum = 0;
  for i=1:num_particles
    Psum = Psum + W(i,T) * res(i,:)' * res(i,:);
  end
  
end
