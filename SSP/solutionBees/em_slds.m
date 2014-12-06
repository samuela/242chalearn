function [ model, x0, P0, ll_iter ] = em_slds(...
  model_init, x0_init, P0_init, seq, data, max_iters, conv_tol)
% EM_SLDS - EM for learning the Switching LDS.  This is a heuristic
%   approximation to the optimal approach.
%
% Brown CS242

  num_states = numel(model_init);
  
  % init stuff  
  done = false;  
  iters = 0;
  ll_old = -Inf;
  ll_iter = [];
  model = model_init;
  x0 = x0_init;
  P0 = P0_init;
  
  D = numel(x0);
  
  % estimate transition probabilities
  T = zeros([num_states,num_states]);
  for s=1:numel(seq)
    thisSeq = seq{s};    
    
    % find transitions
    labels_shift = [ thisSeq.label(1), thisSeq.label(1:(end-1)) ];
    tpoints_end = find( thisSeq.label ~= labels_shift );
    tpoints_start = tpoints_end-1;
    for t_i=1:numel(tpoints_end)
      k_prev = thisSeq.label(tpoints_start(t_i));
      k_next = thisSeq.label(tpoints_end(t_i));
      T( k_prev, k_next ) = T( k_prev, k_next ) + 1;
      if t_i>1
        n = tpoints_start(t_i) - tpoints_start(t_i-1) - 1;
      else
        n = tpoints_start(t_i);
      end
      T( k_prev, k_prev ) = n;
    end    
  end  
  T0 = sum( T, 2 ) ./ sum(T(:));  
  T = T ./ repmat( sum( T, 2 ), [1,num_states] );
  
  % save discrete state params
  for s=1:num_states
    model{s}.T0 = T0;
    model{s}.T = T;
  end
  
  % Main Loop
  while ~done
    iters = iters+1;
    fprintf('\nIter #%d:', iters);
    
    %% E-step %%%%%%%%%%%%%
    
    % init stuff
    Eedge = cell([num_states,1]);
    Exx = cell([num_states,1]);    
    Xs = cell([num_states,1]);    
    Ps = cell([num_states,1]); 
    Nscans = cell([num_states, 1]);
    
    % compute statistics
    for k=1:num_states
      num_seq = numel( data{k} );
      Eedge{k} = cell([num_seq, 1]);
      Exx{k} = cell([num_seq, 1]);
      Xs{k} = cell([num_states,1]);    
      Ps{k} = cell([num_states,1]);    
      Nscans{k} = cell([num_states,1]);
      
      % iterate over sequences
      for i_seq = 1:num_seq
        Nscans{k}{i_seq} = size(data{k}{i_seq},2);
        
        % run Kalman smoother
        [ ~, Pf, Xs{k}{i_seq}, Ps{k}{i_seq} ] = ...
          kalman_smoother(model{k}, data{k}{i_seq}, x0, P0 );
        
      P = zeros( D, D, Nscans{k}{i_seq} );
      P_pred = zeros( D, D, Nscans{k}{i_seq} );
      for t=1:Nscans{k}{i_seq}
          
          % Prediction Step
          if t>1
              P_pred(:,:,t) = model{k}.A*P(:,:,t-1)*model{k}.A' + model{k}.Q;
          else
              P_pred(:,:,t) = P0;
          end
          
          K = P_pred(:,:,t) * model{k}.C' / (model{k}.C*P_pred(:,:,t)*model{k}.C' + model{k}.R);
          P(:,:,t) = (eye(D) - K*model{k}.C)*P_pred(:,:,t);
      end
      
      
        for t=1:Nscans{k}{i_seq}     

          % E[x_t x_t']
          Exx{k}{i_seq}(:,:,t) = Ps{k}{i_seq}(:,:,t) + Xs{k}{i_seq}(:,t) * Xs{k}{i_seq}(:,t)';

          % E[x_t x_{t-1}']
          if t>=2
            C = Pf(:,:,t-1) * model{k}.A' / P_pred(:,:,t);
            Eedge{k}{i_seq}(:,:,t) = ...
              Ps{k}{i_seq}(:,:,t) * C' + Xs{k}{i_seq}(:,t) * Xs{k}{i_seq}(:,t-1)';      
          end
          
          % check conditions
          if rcond(Exx{k}{i_seq}(:,:,t)) <= 1e-9
            fprintf('\nWARNING Matrix Exx{%d}{%d} ill-conditioned at time %d.', k, i_seq, t);
          end
          if rcond(Eedge{k}{i_seq}(:,:,t)) <= 1e-9
            fprintf('\nWARNING Matrix Eedge{%d}{%d} ill-conditioned at time %d.', k, i_seq, t);
          end
        end
      end
    end
    
    % compute free energy
    ll = compute_slds_bound( num_states, model, data, x0, P0, Eedge, Exx, Xs, Ps );
    ll_iter = [ ll_iter; ll ];
    if iters>1
      diff = (ll - ll_old)/abs(ll);
      fprintf('\n(Iter #%d) LL: %0.3f, D: %0.3f', iters, ll, diff);
      if ( ( diff < conv_tol ) || (iters >= max_iters) )
        break;
      end
    end
    ll_old = ll;
    
%     % compute free energy
%     ll_e = compute_slds_bound( num_states, model, data, x0, P0, Eedge, Exx, Xs, Ps );
%     if iters>1
%       diff = ll_e - ll_old;
%       fprintf('\n   E-Step LL: %0.3f, D: %0.4f', ll_e, diff/abs(ll_old));
%     end
    
    %% M-step %%%%%%%%%%%%    
    for k=1:num_states
      A_new = 0; Q_new = 0; Q_new_T = 0; C_new = 0; R_new = 0; R_new_T = 0; 
      
      % compute stats
      Eedge_sum = 0; Exx_sum = 0; Exx_head_sum = 0;
      for i_seq = 1:numel(data{k})
        Eedge_sum = Eedge_sum + sum( Eedge{k}{i_seq}, 3);
        Exx_sum = Exx_sum + sum( Exx{k}{i_seq}, 3 );
        Exx_head_sum = Exx_head_sum + sum( Exx{k}{i_seq}(:,:,1:(end-1)), 3);
      end

      % dynamics
      A_new = A_new + Eedge_sum / Exx_head_sum;
      for i_seq = 1:numel(data{k})
        for t=2:Nscans{k}{i_seq}
          Q_new = Q_new + Exx{k}{i_seq}(:,:,t) - A_new * Eedge{k}{i_seq}(:,:,t)' ...
            - Eedge{k}{i_seq}(:,:,t) * A_new' + A_new * Exx{k}{i_seq}(:,:,t-1) * A_new';
        end        
      end
      Q_new = 1/(sum(cell2mat(Nscans{k}))-1) * Q_new;
      for i_seq = 1:numel(data{k})
        for t=2:Nscans{k}{i_seq}
          Q_new_T = Q_new_T + Exx{k}{i_seq}(:,:,t)' - Eedge{k}{i_seq}(:,:,t) * A_new'...
            - A_new*Eedge{k}{i_seq}(:,:,t)' + A_new * Exx{k}{i_seq}(:,:,t-1)' * A_new';
        end        
      end
      Q_new_T = 1/(sum(cell2mat(Nscans{k}))-1) * Q_new_T;
      Q_new = 0.5 * ( Q_new + Q_new_T );

      % observation
%       for i_seq = 1:numel(data{k})
%         for t=1:Nscans{k}{i_seq}
%           H_new = H_new + data{k}{i_seq}(t).Z * Xs{k}{i_seq}(:,t)';
%         end
%       end
%       H_new = H_new / Exx_sum ;
      C_new = model{k}.C;
      for i_seq = 1:numel(data{k})
        for t=1:Nscans{k}{i_seq}
          R_new = R_new + data{k}{i_seq}(:,t) * data{k}{i_seq}(:,t)' ...
            - C_new * Xs{k}{i_seq}(:,t) * data{k}{i_seq}(:,t)' ...
            - data{k}{i_seq}(:,t) * Xs{k}{i_seq}(:,t)' * C_new' ...
            + C_new * Exx{k}{i_seq}(:,:,t) * C_new';
        end
      end
      R_new = 1/sum(cell2mat(Nscans{k})) * R_new;
      for i_seq = 1:numel(data{k})
        for t=1:Nscans{k}{i_seq}
          R_new_T = R_new_T + data{k}{i_seq}(:,t) * data{k}{i_seq}(:,t)' ...
            - C_new * Xs{k}{i_seq}(:,t) * data{k}{i_seq}(:,t)' ...
            - data{k}{i_seq}(:,t) * Xs{k}{i_seq}(:,t)' * C_new' ...
            + C_new * Exx{k}{i_seq}(:,:,t)' * C_new';
        end
      end
      R_new_T = 1/sum(cell2mat(Nscans{k})) * R_new_T;      
      R_new = 0.5 * ( R_new + R_new_T );
      
%       % make cov mats symmetric
%       L_q = chol(Q_new);
%       Q_new = L_q' * L_q;
%       L_r = chol(R_new);
%       R_new = L_r' * L_r;

      % update model
      model{k}.A = A_new;
      model{k}.Q = Q_new;
      model{k}.C = C_new;
      model{k}.R = R_new;
    
    end
    
%     % compute free energy
%     ll = compute_slds_bound( num_states, model, data, x0, P0, Eedge, Exx, Xs, Ps );
%     if iters>1
%       diff = ll - ll_old;
%       fprintf('\n   M-Step LL: %0.3f, D: %0.4f', ll, diff/abs(ll_old));
%       if ( diff>=0 ) && ( (diff/abs(ll_old)) < conv_tol )
%         done = true;
%       elseif diff < 0
%         fprintf('\nLikelihood decreased!');
%       end
%       if iters == max_iters, done = true; end;
%     end
%     ll_old = ll;
%     
%     ll_iter = [ ll_iter; ll ];
  end
  
  for(k=1:numel(model))
     model{k}.Q = (model{k}.Q + model{k}.Q')/2;
  end
  
end
