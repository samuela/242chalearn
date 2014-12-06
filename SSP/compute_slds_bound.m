function ll = compute_slds_bound( num_states, model, data, x0, P0, Eedge, Exx, Xs, Ps )
% COMPUTE_LDS_BOUND - Computes the EM lower bound ("auxilliary function")
%   for the linear dynamical system with model parameters 'model'.
%
% Brown CS242
  
  % loop over states
  ll = 0;
  for k=1:num_states
    
    % unpack alphabet
    [A, C, Q, R ] = deal( model{k}.A, model{k}.C, model{k}.Q, model{k}.R );
    Qinv = inv(Q);
    Rinv = inv(R);
    P0inv = inv(P0);    
    
    % loop over sequences
    num_seq = numel(data{k});
    for i_seq = 1:num_seq      
      T = size(data{k}{i_seq},2);

      % compute bound
      for t=1:T
        if t==1
          ll = ll -1/2*trace(P0inv*Exx{k}{i_seq}(:,:,1)) ...
            + Xs{k}{i_seq}(:,1)'*P0inv*x0 - 1/2*x0'*P0inv*x0;
        else
          ll = ll - 1/2 * trace(Qinv * Exx{k}{i_seq}(:,:,t)) + trace(Qinv*A*Eedge{k}{i_seq}(:,:,t)') ...
            - 1/2 * trace(A'*Qinv*A * Exx{k}{i_seq}(:,:,t-1));
        end
        y = data{k}{i_seq}(:,t);
        ll = ll - 1/2 * y'*Rinv*y + y'*Rinv*C*Xs{k}{i_seq}(:,t) ...
          - 1/2 * trace(C'*Rinv*C*Exx{k}{i_seq}(:,:,t));
      end    

      % add entropy
      for t=1:T
        ll = ll + 1/2*log( det( Ps{k}{i_seq}(:,:,t) ) );
      end
      
      % normalizers
      ll = ll - 1/2 * log( det(P0) ) - (T-1)/2 * log( det( Q ) ) - T/2 * log( det( R) );
    end
  end
end
