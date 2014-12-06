function [ Xf, Pf, Xs, Ps ] = kalman_smoother(model, data, x0, P0)
% Inputs:
%    model: the data model structure, containing the fields A,C,Q,R, as
%           defined in the handout
%    data: D x T vector containing the time-series data
%    x0: guess for the initial state
%    P0: covariance matrix on the initial state
%
% Outputs:
%   Xf: Kalman-Filter posterior mean estimate for the state at time t
%   Pf: Kalman-Filter posterior variance estimate for the state at time t
%
%   Xs: Kalman-Smoother posterior mean estimate for the state at time t
%   Ps: Kalman-smoother posterior variance estimate for the state at time t
%
% Brown CS242

  T = size(data,2);
  
  % init estimates
  D = numel(x0);
  x = zeros( D, T );
  P = zeros( D, D, T );
  P_pred = zeros( D, D, T );
  x_pred = zeros( D, T );
  
  % unpack alphabet
  [ A, C, Q, R ] = deal( model.A, model.C, model.Q, model.R );

  % forward pass
  for t=1:T
    
    % Prediction Step
    if t>1
      x_pred(:,t) = A*x(:,t-1);
      P_pred(:,:,t) = A*P(:,:,t-1)*A' + Q;
    else
      x_pred(:,t) = x0;
      P_pred(:,:,t) = P0;
    end
    
    % Time update
    %z_i = data(t).Z;
    z_i = data(:,t);
    S = C*P_pred(:,:,t)*C' + R;
    K = P_pred(:,:,t) * C' / S;
    nu = z_i - C * x_pred(:,t);
    x(:,t) = x_pred(:,t) + K * nu;
    P(:,:,t) = (eye(D) - K*C)*P_pred(:,:,t);
    
  end
  Xf = x;
  Pf = P;
  
  % reuse moments
  x = zeros(D,T);
  P = zeros(D,D,T);
  
  % backward pass 
  x(:,T) = Xf(:,T);
  P(:,:,T) = Pf(:,:,T);
  for t=(T-1):-1:1
    C = Pf(:,:,t) * A' / P_pred(:,:,t+1);
    x(:,t) = Xf(:,t) + C * ( x(:,t+1) - x_pred(:,t+1) );
    P(:,:,t) = Pf(:,:,t) + C * ( P(:,:,t+1) - P_pred(:,:,t+1) ) * C';
  end  
  
  Xs = x;
  Ps = P;
end
