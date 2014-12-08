function [Xpost, Zpost] = particle_filter_sam(num_particles, model, data, pi)
% Inputs:
%    num_particles: number of particles to use
%    model: a cell array containing structs with A, C, Q, R for each object category.
%    data: D x T vector containing the time-series data
%    pi: transition probability matrix for z_t's
% Outputs:
%    Xpost: D x T matrix containing the D-dimensional posterior mean of the
%           estimate for the states from time t=1 to t=T
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
    % Re-weight particles
    w = zeros(num_particles, 1);
    
%     %slow way
%     for i=1:num_particles
%       w(i) = mvnpdf(data(:,t), model{Z(i)}.C * X(:,i), model{Z(i)}.R);
%     end
%     w = w/sum(w);

    %fast way
    for k=1:num_states
%       (model{k}.C * X(:,Z == k))'
      if sum(Z == k) > 0
%           data(:,t)';
%           mvnpdf(data(:,t)', ...
%                                  (model{k}.C * X(:,Z == k))', ...
%                                  model{k}.R + 0.1 * eye(D));
        %fprintf('state %d\n',k);
        cov_mat = model{k}.R;
        cov_mat = (cov_mat + cov_mat')/2; %its very close to symmetric, but not exactly.
%         eig(cov_mat)
%         if ~all(eig(cov_mat) > 0)
%                 [V, d] = eig(cov_mat);
%                 d1 = sum(d, 2);
%                 d1 = max(d1, .001);
%                 cov_mat = V * diag(d1) / V;
%         end
%        eig(cov_mat)
%        eig_tol = 0.00001;
%        [V, eig_diag] = eig(cov_mat);
%        eig_diag = sum(eig_diag); %collapse it from a diagonal matrix to a vector
%        if any(eig_diag <= eig_tol)
%          cov_mat = V * diag(max(eig_diag, eig_tol)) / V;
%        end
%        eig(cov_mat)
        w(Z == k) = logmvnpdf(data(:,t)', ...
                                 (model{k}.C * X(:,Z == k))', ...
                                 cov_mat);
      end
    end
    m = mean(w);
    normalizing = m+log(sum(exp(w-m)));
    w = exp(w - normalizing);
    
    % Calculate posterior mean and Z_t distribution
    Xpost(:,t) = X * w;
    temp = zeros(num_states, 1);%[(Z == 1) * w; (Z == 2) * w; (Z == 3) * w];
    for state_num = 1:num_states
        temp(state_num) = (Z == state_num) * w;
    end
    Zpost(:,t) = temp / sum(temp);
    
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
        newZ(Z == k) = randsample(1:num_states, sum(Z == k), true, pi(k,:));
      end
    end
    Z = newZ;
    
    newX = zeros(D, num_particles);
    for k=1:num_states
      if sum(Z == k) > 0
        eig_tol = 0.00001;
        cov_mat = model{k}.Q;
        [V, eig_diag] = eig(cov_mat);
        eig_diag = sum(eig_diag); %collapse it from a diagonal matrix to a vector
        if any(eig_diag <= eig_tol)
          cov_mat = V * diag(max(eig_diag, eig_tol)) / V;
        end
        
        newX(:,Z == k) = mvnrnd((model{k}.A * X(:,Z == k))', cov_mat, sum(Z == k))';
      end
    end
    X = newX;
  end
end
