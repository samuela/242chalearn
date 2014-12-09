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

    %fast way
    for k=1:num_states
      if sum(Z == k) > 0
        cov_mat = model{k}.R;
        cov_mat = (cov_mat + cov_mat')/2; 
        w(Z == k) = logmvnpdf(data(:,t)', ...
                                 (model{k}.C * X(:,Z == k))', ...
                                 cov_mat);
      end
    end
    
    % weighting scheme
    % find some constant c such that exp^log(weights) ~= 0
    % and then you know you found the real weights,
    % scooting around any numerical issues
    temp = zeros(size(w));
    w_sort = sort(w); ii = 1;
    while all(temp==0)+any(isnan(temp)) == 1
        temp = exp(w-w_sort(ii));
        temp = temp/sum(temp);
        ii = ii+1;
    end
    w = temp;
    
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
