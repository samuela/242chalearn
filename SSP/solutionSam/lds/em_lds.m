function [ model, ll_iter ] = em_lds(model_init, x0, P0, data, max_iters, conv_tol)
% EM_LDS - Expectation Maximization for learning the Linear Dynamical
% Inputs: 
%    model_init: Initial model. Contains fields A,C,Q,R which correspond
%                to the parameters defined in Question 1 of the handout.
%    x0: fixed mean of initial state
%    P0: fixed covariance of initial state
%    data: T x D matrix containing the D-dimensional observed data for each
%          t=1:T time step
%    max_iters: Maximum iterations to run EM for
%    conv_tol: Convergence threshhold for log-likelihood
% Outputs:
%    model: the estimated. Must contain fields A,C,Q,R.
%    ll_iter: marginal log-likelihood at each iteration
%
% Brown CS242

  % init stuff
  T = size(data,1);
  done = false;  
  iters = 0;
  ll_old = -Inf;
  model = model_init;
  D = numel(x0);
  ll_iter = [];
  
  % Main Loop
  while ~done
    iters = iters+1;
%     fprintf('\nIter #%d:', iters);
    
    %% E-step %%%%%%%%%%%%%
    
    % The Kalman-smoother is used to compute the sufficient statistics 
    % that you'll need to perform the M-step parameter updates.
    [Xf, Pf, Xs, Ps] = kalman_smoother(model, data', x0, P0);
    
    % Various quantities useful for the M-step can be computed from
    % the Kalman filter and smoother outputs.  Here are some examples:
    Eedge = zeros([D,D,T]);
    Exx = zeros([D,D,T]);
    for t=1:T
      if t > 1
        P_pred = model.A*Pf(:,:,t-1)*model.A' + model.Q;
      else
        P_pred = P0;
      end
      
      % E[x_t x_t' | y]
      Exx(:,:,t) = Ps(:,:,t) + Xs(:,t) * Xs(:,t)';
      
      % E[x_t x_{t-1}' | y]
      if t >= 2
        C = Pf(:,:,t-1) * model.A' / P_pred;
        Eedge(:,:,t) = Ps(:,:,t) * C' + Xs(:,t) * Xs(:,t-1)';      
      end
    end
    
    [A, Q, C, R] = deal(model.A, model.Q, model.C, model.R);
    
    % FILL IN THE COMPUTATION OF THE MARGINAL LOG-LIKELIHOOD AT THIS
    % ITERATION. CODE BELOW IS JUST A PLACE-HOLDER FOR YOUR CODE.
    ll = logmvnpdf(data(1,:)', C * x0, R + C * P0 * C');
    
    for t=1:(T-1)
      ll = ll + logmvnpdf(data(t+1,:)', C * A * Xf(:,t), ...
                          C*Q*C' + R + (C*A) * Pf(:,:,t) * (C*A)');
    end
    
    ll_iter(end+1) = ll;
    if iters > 1
      diff = (ll - ll_old) / abs(ll);
      fprintf('\n(Iter #%d) LL: %0.3f, D: %0.3f', iters, ll, diff);
      if ( ( diff < conv_tol ) || (iters >= max_iters) )
%       if iters >= max_iters
        break;
      end
    end
    ll_old = ll;
    
    %% M-step %%%%%%%%%%%%
     
    % we provide the updates for A and Q
    A_new = sum( Eedge, 3 ) / sum(Exx(:,:,1:(end-1)), 3);
    Q_new = 0;
    for t=2:T
      Q_new = Q_new + Exx(:,:,t) - A_new * Eedge(:,:,t)' - Eedge(:,:,t) * A_new' ...
        + A_new * Exx(:,:,t-1) * A_new';
    end
    Q_new = 1/(T-1) * Q_new;
    
    % no update for C; it's always [I_p, 0];
    C_new = model.C;
    
    % PLACEHOLDER CODE. REPLACE WITH YOUR M-STEP CODE to update parameters
    % R of the model.
    R_new = zeros(size(R));
    for t=1:T
      R_new = R_new + data(t,:)' * data(t,:) - data(t,:)' * Xs(:,t)' * C_new';
    end
    
    % update model
    model.A = A_new;
    model.Q = Q_new;
    model.C = C_new;
    model.R = R_new / T;
  
    %for numerical reasons
    model.Q = (model.Q + model.Q')/2;
    model.R = (model.R + model.R')/2;
  end
  
end
