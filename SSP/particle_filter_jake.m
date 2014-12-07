function [X, Z] = particle_filter_jake(L, model, data, Pi, corruption_prob, corruption_var)
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
%    Z: K x T
% Brown CS242

  if nargin < 5
    corruption_prob = 0;
    corruption_var  = 1;
  end
  
  D = size(data,1);
  T = size(data,2);
  X = zeros(D,T);
  K = length(model);
  Z = zeros(K,T);


  part_x = mvnrnd(zeros(D,1),eye(D),L);
  part_z = randsample(K,L,true);
  w = ones(1,L)/L;
  
  % unpack model
  [A, C, Q, R] = deal(cell(1,K));
  for kk=1:K
      [A{kk},C{kk},Q{kk},R{kk}] = deal(model{kk}.A, model{kk}.C, model{kk}.Q, model{kk}.R );
  end
  
  
% slow way
  
%{
  for t=1:T
    t/T
    % weight by observation likelihood
    for l=1:L
        kk = part_z(l);
        w(l) = ((1-corruption_prob)*mvnpdf(data(:,t)',(C{kk}*part_x(l,:)')',corruption_prob*eye(D)+R{kk})' + ...
              corruption_prob*mvnpdf(data(:,t)',zeros(1,D),corruption_var*eye(D))');
    end
    w = w/sum(w);

    % mean posterior estimate at time t
    X(:,t) = part_x'*w';
    for kk=1:K
        Z(kk,t) = sum(w(find(part_z==kk)));
    end
    Z(:,t) = Z(:,t)/sum(Z(:,t));
    
    % resample & propagate by dynamics (model)
    rsmpl_ind = randsample(L,L,true,w);
    part_x = part_x(rsmpl_ind,:); 
    part_z = part_z(rsmpl_ind,:);
    
    %{
    pi_est = zeros(K);
    labels = part_z;
    for ii=1:K
        states = labels(find(labels(1:end-1)==ii)+1);
        for jj=1:K
            pi_est(ii,jj)=pi_est(ii,jj)+sum(states==jj);
        end
    end
    pi_est = pi_est./repmat(sum(pi_est,2),1,K);
    %}
    
    for l=1:L
        kk = part_z(l);
        % propagate via p(z_t | z_{t-1}=k) = Cat(z_t | hat{pi}_k)
        part_z(l) = randsample(K,1,true,Pi(kk,:));
        % propagate via p(x_t | x_{t-1}, z_t = k) = Norm(x_t | A_k*x_{t-1},Q_k)
        part_x(l,:) = mvnrnd((A{kk}*part_x(l,:)')',Q{kk});
    end
  end
%}
  for  t=1:T
    100*t/T
    for kk=1:K
        inds = find(part_z==kk);
        w(inds) = ((1-corruption_prob)*mvnpdf(data(:,t)',(C{kk}*part_x(inds,:)')',corruption_prob*eye(D)+R{kk})' + ...
            corruption_prob*mvnpdf(data(:,t)',zeros(1,D),corruption_var*eye(D))');
    end
    w = w/sum(w);

    % mean posterior estimate at time t
    X(:,t) = part_x'*w';
    for kk=1:K
        Z(kk,t) = sum(w(find(part_z==kk)));
    end
    Z(:,t) = Z(:,t)/sum(Z(:,t));

    
    % resample & propagate by dynamics (model)
    rsmpl_ind = randsample(L,L,true,w);
    part_x = part_x(rsmpl_ind,:); 
    part_z = part_z(rsmpl_ind,:);

    for kk=1:K
        inds = find(part_z==kk);
        if numel(inds)>0
            % propagate via p(z_t | z_{t-1}=k) = Cat(z_t | hat{pi}_k)
            part_z(inds) = randsample(K,numel(inds),true,Pi(kk,:));
            % propagate via p(x_t | x_{t-1}, z_t = k) = Norm(x_t | A_k*x_{t-1},Q_k)
            part_x(inds,:) = mvnrnd((A{kk}*part_x(inds,:)')',Q{kk});
        end
    end

      
  end

end