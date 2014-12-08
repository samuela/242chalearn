clear variables;
num_states = 21;

    %% Get data
    fprintf('...Gathering data\n')
    inds = [1:11];
    inds(1) = [];
    % split into train/test
    [labels_train, data_train, ~] = load_data('../../data/skainswo/chalearn/train', inds);
    [labels_test, ~, seqs_test] = load_data('../../data/skainswo/chalearn/train', 1);
    labels_test = labels_test{1};
    seqs_test = seqs_test{1};
    
    %% Transition Probs
    % Compute ML estimates of the transition probabilities in the training set.
    % Compute using the data in seq_train.
    fprintf('...Computing transition probabilities\n');
    trans_est = zeros(num_states);
    for kk=1:numel(labels_train)
        labels = labels_train{kk}; 
        for ii=1:num_states
            states = labels(find(labels(1:end-1)==ii)+1);
            for jj=1:num_states
                trans_est(ii,jj)=trans_est(ii,jj)+sum(states==jj);
            end
        end
    end
    trans_est = trans_est./repmat(sum(trans_est,2),1,num_states);
    
    %% Run EM learning
    fprintf('...Starting EM\n');
    
    conv_tol = 1e-4;
    max_iters = 8;
    [model_init, x0_init, P0_init ] = init_model(num_states,size(data_train{1}{1},1),size(data_train{1}{1},1)); % "7*<num_joints>" depends on how many joints we use
    model_est = cell([num_states,1]);
    ll_iter = cell([num_states,1]);
    
%     parfor kk = 1:num_states
%         transposed_data = cellfun(@(x)x',data_train{kk},'UniformOutput',false);
%         [ model_est{kk}, ll_iter{kk} ] = em_lds_general(model_init{kk}, x0_init, P0_init, transposed_data, max_iters, conv_tol);
%         fprintf('trained %d',kk);
%     end

    [ model_est, x0_est, P0_est, ll_iter ] = ...
      em_slds(model_init, x0_init, P0_init, { labels_test }, data_train, max_iters, conv_tol);

    %% Plot
  
%     fig = figure(); hold on
%     for kk=2:num_states
%         subplot(5,4,kk-1)
%         plot( ll_iter(kk,:), '-b', 'LineWidth', 3 );
%         %axis([1 numel(ll_iter{kk}) ll_iter{kk}(1)-10 ll_iter{kk}(end)+1000])
%     end
    fig = figure();
    plot( ll_iter, '-b', 'LineWidth', 3 );
    xlabel('Iteration #');
    ylabel('Log-Likelihood');
    saveas(fig, 'logLikeTA.pdf');
    
    
    %% Particle filtering
    fprintf('...Start particle filtering\n');
    num_particles = 100;
    [X_est, Z_est] = particle_filter_sam(num_particles, model_est, seqs_test, trans_est);
    %can also try particle_filter_sam, gets the same error

    
    