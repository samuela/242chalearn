function [ model, ll_iter] = em_lds_general(model_init, x0, P0, data, max_iters, conv_tol)
% EM_LDS - Expectation Maximization for learning the Linear Dynamical
% Inputs: 
%    model_init: Initial model. Contains fields A,C,Q,R which correspond
%                to the parameters defined in Question 1 of the handout.
%    x0: fixed mean of initial state
%    P0: fixed covariance of initial state
%    data: cell array of T_s x D matrices containing the D-dimensional 
%          observed data for each t=1:T_s time step for each segment s
%    max_iters: Maximum iterations to run EM for
%    conv_tol: Convergence threshhold for log-likelihood
% Outputs:
%    model: the estimated. Must contain fields A,C,Q,R.
%    ll_iter: marginal log-likelihood at each iteration
%
% Brown CS242


  % general init
  done = false;
  conv_for = 0; % # iters converged for
  iters = 0;
  ll_old = -Inf;
  model = model_init;  
  S = numel(data);
  D = numel(x0);
  ll_iter = [];
  T_total = sum(cellfun(@length,data));
  T=zeros(S,1);
  for s=1:S
    T(s) = size(data{s},1);
  end
  % Main Loop
  while ~done
    iters = iters+1;
    %% E-step %%%%%%%%%%%%%
    ll=0;
    [Eedge, Exx]=deal(cellfun(@(seg) zeros(D,D,size(seg,1)),data,'un',0));
    [Xf, Pf, Xs, Ps ] = deal(cell(S,1));
    % unpack alphabet
    [A, C, Q, R] = deal(model.A, model.C, model.Q, model.R);
    for s=1:S
        % The Kalman-smoother is used to compute the sufficient statistics 
        % that you'll need to perform the M-step parameter updates.
        [Xf{s}, Pf{s}, Xs{s}, Ps{s}] = kalman_smoother(model, data{s}', x0, P0);
        for t=1:T(s)      
          if (t > 1)
            P_pred = A*Pf{s}(:,:,t-1)*A' + Q;
          else
            P_pred = P0;
          end
      
          % E[x_t x_t' | y]
          Exx{s}(:,:,t) = Ps{s}(:,:,t) + Xs{s}(:,t) * Xs{s}(:,t)';
          if rcond(Exx{s}(:,:,t)) < 10^(-22)
            break;
          end          
              
          % E[x_t x_{t-1}' | y]
          if t>=2
            temp = Pf{s}(:,:,t-1) * A' / P_pred;
            Eedge{s}(:,:,t) = Ps{s}(:,:,t) * temp' + Xs{s}(:,t) * Xs{s}(:,t-1)';      
            if rcond(Eedge{s}(:,:,t)) < 10^(-22)
                break;
            end
          end
        end
    
        
        
        
        
        % MARGINAL LOG-LIKELIHOOD
        ll = ll + log(mvnpdf(data{s}(1,:)',C*x0,C*P0*C'+R));
        for t=1:(T(s)-1)
            covv = (C*A)*Pf{s}(:,:,t)*(C*A)'+C*Q*C'+R;
            %[v,d] = eig(covv);
            %covv = v*max(0.001,D)/v 
            covv = (covv + covv')/2;
            if ~all(eig(covv)>0)
                %covv = covv + eye(size(covv))*(-2*min(eig(covv)));
                %covv
                [vv,dd] = eig(covv);  
                covv = vv*(eye(size(dd)).*max(dd,0.0001))/vv;
            end
            covv = (covv + covv')/2;
            ll = ll + log(mvnpdf(data{s}(t+1,:)',C*A*Xf{s}(:,t),covv));
        end
        %{
        
        Qinv = inv(Q);
        Rinv = inv(R);
        P0inv = inv(P0);    
        
        for t=1:T(s)
          if t==1
            ll = ll -1/2*trace(P0inv*Exx{s}(:,:,1)) ...
                + Xs{s}(:,1)'*P0inv*x0 - 1/2*x0'*P0inv*x0;
          else
            ll = ll - 1/2 * trace(Qinv * Exx{s}(:,:,t)) ...
                + trace(Qinv*A*Eedge{s}(:,:,t)') ...
                - 1/2 * trace(A'*Qinv*A * Exx{s}(:,:,t-1));
          end
          y = data{s}(t,:)';
          ll = ll - 1/2 * y'*Rinv*y + y'*Rinv*C*Xs{s}(:,t) ...
             - 1/2 * trace(C'*Rinv*C*Exx{s}(:,:,t));
        end
            
        for t=1:T(s)
             ll = ll + 1/2*log( det( Ps{s}(:,:,t) ) );
        end
        ll = ll - 1/2 * log( det(P0) ) - (T-1)/2 * log( det( Q ) ) ...
              - T/2 * log( det( R) );
        %}

    end
    ll=sum(ll(:));

    if ~isfinite(ll) %|| ll < ll_old
        fprintf('\ndone cuz infinite ll');
        break;
    end
    
    ll_iter(end+1) = ll;
    if iters>1
      diff = (ll - ll_old)/abs(ll);
      fprintf('\n(Iter #%d) LL: %0.3f, D: %0.3f', iters, ll, diff);
      conv_for = ( diff < conv_tol )*(conv_for+1);
      if ( conv_for > 3 || (iters >= max_iters) ) % ( ( diff < conv_tol ) || (iters >= max_iters) )
        fprintf('\ndone cuz below tolerance');
        break;
      end
    end
    ll_old = ll;

    %% M-step %%%%%%%%%%%%
%{
    A_new = zeros(D,D);
    Q_new = zeros(D,D);
    R_new = zeros(size(R));
    
    [A_num, A_den] = deal(A_new);
    for s=1:S
        A_num = A_num + sum(Eedge{s}, 3);
        A_den = A_den + sum(Exx{s}(:,:,1:(end-1)), 3);
    end
    A_new = A_num / A_den;
    
    for s=1:S
        %T = size(data{s},1);
        for t=1:T(s)
            if t>1
                Q_new = Q_new + Exx{s}(:,:,t) - A_new * Eedge{s}(:,:,t)' - Eedge{s}(:,:,t) * A_new' ...
                        + A_new * Exx{s}(:,:,t-1) * A_new';
            end
            R_new = R_new + data{s}(t,:)'*data{s}(t,:) - data{s}(t,:)'*Xs{s}(:,t)'*C';
        end
    end
    Q_new = Q_new / (T_total-S);
    Q_new = (Q_new + Q_new') / 2;
    R_new = R_new / T_total;
    R_new = (R_new + R_new') / 2;
        
    % update model
    model.A = A_new;
    model.Q = Q_new;
    model.R = R_new;
%}
%% M-step %%%%%%%%%%%%    
    A_new = 0; Q_new = 0; Q_new_T = 0; C_new = 0; R_new = 0; R_new_T = 0; 
      
    % compute stats
    Eedge_sum = 0; Exx_sum = 0; Exx_head_sum = 0;
    for s = 1:S
        Eedge_sum = Eedge_sum + sum( Eedge{s}, 3);
        Exx_sum = Exx_sum + sum( Exx{s}, 3 );
        Exx_head_sum = Exx_head_sum + sum( Exx{s}(:,:,1:(end-1)), 3);
    end

    %  dynamics
    A_new = A_new + Eedge_sum / Exx_head_sum;
    for s = 1:S
        for t=2:T(s)
              Q_new = Q_new + Exx{s}(:,:,t) - A_new * Eedge{s}(:,:,t)' ...
                - Eedge{s}(:,:,t) * A_new' + A_new * Exx{s}(:,:,t-1) * A_new';
        end        
    end
    Q_new = 1/(sum(T)-1) * Q_new;
    for s = S
        for t=2:T(s)
              Q_new_T = Q_new_T + Exx{s}(:,:,t)' - Eedge{s}(:,:,t) * A_new'...
                - A_new*Eedge{s}(:,:,t)' + A_new * Exx{s}(:,:,t-1)' * A_new';
        end        
    end
    Q_new_T = 1/(sum(T)-1) * Q_new_T;
    Q_new = 0.5 * ( Q_new + Q_new_T );
    Q_new = (Q_new + Q_new)'/2;
    C_new = model.C;

    
    for s = 1:S
        for t=1:T(s)
              %size(data{s}(t,:)' * data{s}(t,:))
              %size(C_new * Xs{s}(:,t) * data{s}(t,:))
              %size(data{s}(t,:) * Xs{s}(:,t)' * C_new')
              %size(C_new * Exx{s}(:,:,t) * C_new')
              %size(data{s}(:,t))  % 47  x 1
              %size(C_new)         % 140 x 140
              %size(Xs{s}(:,t))    % 140 x 1
              %size(Exx{s}(:,:,t)) % 140 x 140
            R_new = R_new + data{s}(t,:)' * data{s}(t,:) ...
                - C_new * Xs{s}(:,t) * data{s}(t,:) ...
                - data{s}(t,:)' * Xs{s}(:,t)' * C_new' ...
                + C_new * Exx{s}(:,:,t) * C_new';
            %R_new = R_new + data{s}(:,t)' * data{s}(:,t) ...
            %    - C_new * Xs{s}(:,t) * data{s}(:,t)' ...
            %    - data{s}(:,t) * Xs{s}(:,t)' * C_new' ...
            %    + C_new * Exx{s}(:,:,t) * C_new';
        end
    end
    R_new = 1/sum(T) * R_new;
    for s = 1:S
        for t=1:T(s)
              R_new_T = R_new_T + data{s}(t,:)' * data{s}(t,:) ...
                - C_new * Xs{s}(:,t) * data{s}(t,:) ...
                - data{s}(t,:)' * Xs{s}(:,t)' * C_new' ...
                + C_new * Exx{s}(:,:,t)' * C_new';
        end
    end
    R_new_T = 1/sum(T) * R_new_T;      
    R_new = 0.5 * ( R_new + R_new_T );
    
    %{
    for s=1:S
        for t=1:T(s)
              R_new = R_new + data{s}(t,:)'*data{s}(t,:) ...
                - data{s}(t,:)'*Xs{s}(:,t)'*C_new';
        end
    end
    R_new = R_new / T_total;
    R_new = (R_new + R_new') / 2;
    %}

    % update model
    model.A = A_new;
    model.Q = Q_new;
    model.C = C_new;
    model.R = R_new;
    
end


model.A;
model.Q;
model.C;
model.R;

end
