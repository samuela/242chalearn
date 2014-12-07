function [ model, ll_iter ] = em_lds_general(model_init, x0, P0, data, max_iters, conv_tol)
% EM_LDS - Expectation Maximization for learning the Linear Dynamical
% Inputs:
%    model_init: Initial model. Contains fields A,C,Q,R which correspond
%                to the parameters defined in Question 1 of the handout.
%    x0: fixed mean of initial state
%    P0: fixed covariance of initial state
%    data: cell array of D x T_s matrices containing the D-dimensional
%            observed data for each t=1:T time step
%    max_iters: Maximum iterations to run EM for
%    conv_tol: Convergence threshold for log-likelihood
% Outputs:
%    model: the estimated. Must contain fields A,C,Q,R.
%    ll_iter: marginal log-likelihood at each iteration
%
% Brown CS242

  % init stuff
  S = numel(data);
%   T = size(data,1);
  done = false;
  iters = 0;
  ll_old = -Inf;
  model = model_init;
  D = numel(x0);
  ll_iter = [];
  
  eig_tol = 0.00001;

  % Main Loop
  while ~done
    iters = iters + 1;
%     fprintf('\nIter #%d:', iters);

    %% E-step %%%%%%%%%%%%%
    ll = 0;

    [A, Q, C, R] = deal(model.A, model.Q, model.C, model.R);
    [Xf, Pf, Xs, Ps] = deal(cell(S, 1));
    [Eedge, Exx] = deal(cell(S, 1));
    for s=1:S
      % The Kalman-smoother is used to compute the sufficient statistics
      % that you'll need to perform the M-step parameter updates.
      [Xf{s}, Pf{s}, Xs{s}, Ps{s}] = kalman_smoother(model, data{s}, x0, P0);

      % Various quantities useful for the M-step can be computed from
      % the Kalman filter and smoother outputs.  Here are some examples:
      T_s = size(data{s}, 2);
      Eedge{s} = zeros([D,D,T_s]);
      Exx{s} = zeros([D,D,T_s]);
      for t=1:T_s
        if t > 1
          P_pred = A * Pf{s}(:,:,t-1) * A' + Q;
        else
          P_pred = P0;
        end

        % E[x_t x_t' | y]
        Exx{s}(:,:,t) = Ps{s}(:,:,t) + Xs{s}(:,t) * Xs{s}(:,t)';

        % E[x_t x_{t-1}' | y]
        if t >= 2
          CC = Pf{s}(:,:,t-1) * A' / P_pred;
          Eedge{s}(:,:,t) = Ps{s}(:,:,t) * CC' + Xs{s}(:,t) * Xs{s}(:,t-1)';
        end
      end

      % Calculate the log-likelihood of this segment and add it to the
      % overall log-likelihood, ll.
      ll = ll + logmvnpdf(data{s}(:,1), C * x0, R + C * P0 * C');
      for t=1:(T_s-1)
        cov = C * Q * C' + R + (C * A) * Pf{s}(:,:,t) * (C * A)';
        [V, eig_diag] = eig(cov);
        eig_diag = sum(eig_diag);
        if any(eig_diag <= eig_tol)
          cov = V * diag(max(eig_diag, eig_tol)) / V;
        end
        ll = ll + logmvnpdf(data{s}(:,t+1), C * A * Xf{s}(:,t), cov);
      end
    end

    ll_iter(end+1) = ll;
    if iters > 1
      diff = (ll - ll_old) / abs(ll);
%       fprintf('\n(Iter #%d) LL: %0.3f, D: %0.3f', iters, ll, diff);
      if ( ( diff < conv_tol ) || (iters >= max_iters) )
%       if (iters >= max_iters)
        break;
      end
    end
    ll_old = ll;

    %% M-step %%%%%%%%%%%%

    % we provide the updates for A and Q
    A_new = zeros(D);
    A_new_denom = zeros(D);
    for s=1:S
      A_new = A_new + sum(Eedge{s}, 3);
      A_new_denom = A_new_denom + sum(Exx{s}(:,:,1:(end-1)), 3);
    end
    A_new = A_new / A_new_denom;

    Q_new = zeros(D);
    Q_new_denom = 0;
    for s=1:S
      T_s = size(data{s}, 2);
      for t=2:T_s
        Q_new = Q_new + Exx{s}(:,:,t) - A_new * Eedge{s}(:,:,t)' ...
          - Eedge{s}(:,:,t) * A_new' + A_new * Exx{s}(:,:,t-1) * A_new';
      end
      Q_new_denom = Q_new_denom + (T_s - 1);
%       Q_new_denom = Q_new_denom + T_s;
    end
    Q_new = Q_new / Q_new_denom;

    % no update for C; it's always [I_p, 0];
    C_new = model.C;

    % PLACEHOLDER CODE. REPLACE WITH YOUR M-STEP CODE to update parameters
    % R of the model.
    R_new = zeros(D);
    R_new_denom = 0;
    for s=1:S
      T_s = size(data{s}, 2);
      for t=1:T_s
        R_new = R_new + data{s}(:,t) * data{s}(:,t)' - data{s}(:,t) * Xs{s}(:,t)' * C_new';
      end
      R_new_denom = R_new_denom + T_s;
    end

    % update model
    model.A = A_new;
    model.Q = Q_new;
    model.C = C_new;
    model.R = R_new / R_new_denom;

    %for numerical reasons
    model.Q = (model.Q + model.Q')/2;
    model.R = (model.R + model.R')/2;
  end

end
