function [Xpost, Zpost] = particle_filter_noisy(num_particles, model, data, pi)
%function [mu] = particle_filter_noisy(num_particles, model, data, pi)
% Inputs:
%    num_particles: number of particles to use
%    model: the data model structure, containing the fields A,C,Q,R, as
%           defined in the handout
%    data: ? x T vector containing the time-series data
%    pi: transition probability matrix for z_t's
% Outputs:
%    Xpost: D x T matrix containing the D-dimensional posterior mean of the
%           estimate for the states from time t=1 to time=T
%    Zpost: num_states x T matrix containing posterior categorical
%           distributions
%
% Brown CS242

  D = size(model{1}.A, 1); % sort of a hack
  T = size(data, 2);
  num_states = size(pi, 1);
  Xpost = zeros(D, T);
  Zpost = zeros(num_states, T);

  % Z is 1 x num_particles
  Z = randsample(1:num_states, num_particles, true);
  % X is D x num_particles
  X = mvnrnd(zeros(D, 1), eye(D), num_particles)';
  for t=1:T
%     t/T*100

    % Re-weight particles
    w = zeros(num_particles, 1);
    
    % slow way
%     for i=1:num_particles
%       w(i) = 0.9 * mvnpdf(data(:,t), model{Z(i)}.C * X(:,i), ...
%                           model{Z(i)}.R + 0.1 * eye(D));
%     end

    % fast way
    for k=1:num_states
%       (model{k}.C * X(:,Z == k))'
      if sum(Z == k) > 0
%           data(:,t)';
%           mvnpdf(data(:,t)', ...
%                                  (model{k}.C * X(:,Z == k))', ...
%                                  model{k}.R + 0.1 * eye(D));
        w(Z == k) = 0.9 * mvnpdf(data(:,t)', ...
                                 (model{k}.C * X(:,Z == k))', ...
                                 model{k}.R + 0.1 * eye(D));
      end
    end
    w = w + 0.1 * mvnpdf(data(:,t), zeros(D, 1), 5 * eye(D));
    w = w / sum(w);
    
    % Calculate posterior mean and Z_t distribution
    Xpost(:,t) = X * w;
    poop = [(Z == 1) * w; (Z == 2) * w; (Z == 3) * w];
    Zpost(:,t) = poop / sum(poop);
    
    % Resample
    ix = randsample(1:num_particles, num_particles, true, w);
    Z = Z(ix);
    X = X(:,ix);
    
    % Propogate via transition probability
%     for i=1:num_particles
%       newZ = randsample(1:3, 1, true, pi(Z(i),:));
%       newX = mvnrnd(model{newZ}.A * X(:,i), model{newZ}.Q);
%       Z(i) = newZ; X(:,i) = newX;
%     end
    
    newZ = zeros(1, num_particles);
    for k=1:num_states
      if sum(Z == k) > 0
        newZ(Z == k) = randsample(1:3, sum(Z == k), true, pi(k,:));
      end
    end
    Z = newZ;
    
    newX = zeros(D, num_particles);
    for k=1:num_states
      if sum(Z == k) > 0
        eig_tol = 0.00001;
        cov = model{k}.Q;
        [V, eig_diag] = eig(cov);
        eig_diag = sum(eig_diag);
        if any(eig_diag <= eig_tol)
          cov = V * diag(max(eig_diag, eig_tol)) / V;
        end
        
        newX(:,Z == k) = mvnrnd((model{k}.A * X(:,Z == k))', cov, sum(Z == k))';
      end
    end
    X = newX;
  end
end
