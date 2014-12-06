function ll = compute_lds_bound( model, data, x0, P0, Eedge, Exx, Xs, Ps )
% COMPUTE_LDS_BOUND - Computes the EM lower bound ("auxilliary function")
%   for the linear dynamical system with model parameters 'model'.
%
% Brown CS242

  T = size(data,1);
  D = numel(Xs(:,1));
  M = size(data,2);

  % unpack alphabet
  [ A, C, Q, R ] = deal( model.A, model.C, model.Q, model.R );
  Qinv = inv(Q);
  Rinv = inv(R);
  P0inv = inv(P0);
  
  % compute bound
  ll = -1/2 * log( det(P0) ) - (T-1)/2 * log( det( Q ) ) - T/2 * log( det( R) );  
  for t=1:T
    if t==1
      ll = ll -1/2*trace(P0inv*Exx(:,:,1)) + Xs(:,1)'*P0inv*x0 - 1/2*x0'*P0inv*x0;
    else
      ll = ll - 1/2 * trace(Qinv * Exx(:,:,t)) + trace(Qinv*A*Eedge(:,:,t)') ...
        - 1/2 * trace(A'*Qinv*A * Exx(:,:,t-1));
    end
    y = data(t,:)';
    ll = ll - 1/2 * y'*Rinv*y + y'*Rinv*C*Xs(:,t) ...
      - 1/2 * trace(C'*Rinv*C*Exx(:,:,t));
  end    
  
  % add entropy
  for t=1:T
    ll = ll + D/2*log(2*pi*exp(1)) + 1/2*log( det( Ps(:,:,t) ) );
  end
  
  % constants
  ll = ll - D/2 * log(2*pi) - (T-1)*D/2 * log(2*pi) - T*M/2 * log(2*pi);
end
