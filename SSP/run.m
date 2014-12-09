clear variables;
num_states = 21;

    %% Get data
    file_path = './data';% '../../data/skainswo/chalearn/train'; 
    fprintf('...Gathering data\n')
    train_inds = [2:3];
    test_inds = [1];
    % split into train/test
    [labels_train, data_train, ~] = load_data(file_path, train_inds);
    [labels_test, ~, seqs_test] = load_data(file_path, test_inds);

    
    %% Project Data Into BodSpace

    num_features = 22;
    fprintf('...Compressing data\n')
    % pool the training data
    pooled_data = arrayfun(@(kk) [data_train{kk}{:}],1:size(data_train,1),'un',0);
    pooled_data = [pooled_data{:}]';
    % if needed, here's how you center the data
    % pooled_data = pooled_data-repmat(mean(pooled_data,1),[size(pooled_data,1) 1]);
    coeff = pca(pooled_data);
    proj  = coeff(:,1:num_features)';
    
    for kk = 1:num_states
        for ii = 1:length(data_train{kk})
            data_train{kk}{ii} = proj*data_train{kk}{ii};
        end
    end
    
    for ii = 1:length(seqs_test)
        seqs_test{ii} = proj*seqs_test{ii};
    end
    
    
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
    [model_init, x0_init, P0_init ] = init_model(num_states,size(data_train{1}{1},1),size(data_train{1}{1},1)); 

%    model_est = cell([num_states,1]);
%    ll_iter = cell([num_states,1]);   
%     parfor kk = 1:num_states
%         transposed_data = cellfun(@(x)x',data_train{kk},'UniformOutput',false);
%         [ model_est{kk}, ll_iter{kk} ] = em_lds_general(model_init{kk}, x0_init, P0_init, transposed_data, max_iters, conv_tol);
%         fprintf('trained %d',kk);
%     end

    [ model_est, x0_est, P0_est, ll_iter, var_iter ] = ...
      em_slds(model_init, x0_init, P0_init, labels_test, data_train, max_iters, conv_tol);

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
    %saveas(fig, 'logLikeMoreReg.pdf');
    fig2 = figure();
    plot( log(var_iter), '-r', 'LineWidth', 3 );
    xlabel('Iteration #');
    ylabel('Determinant of R-Parameter');
    %saveas(fig2, 'determinantMoreReg.pdf');
    
    

    %% Finding Good Number of Particles
%{
    num_particles = 50;
    
    scores = zeros(2,15);
    parfor num_particles = 1:15
    current_scores = zeros(1,30);
    for ii = 1:30
        
    [X_est, Z_est] = particle_filter_sam(num_particles*20, model_est, seqs_test, trans_est);
    
    %%% Testing Labels
    groupby = @(labels) ...
        [labels(1) labels(find((labels(1:end-1)-labels(2:end))~=0)+1)];
    
    actual_gestures = groupby(labels_test);
    [~,pred_gestures] = max(Z_est);
    pred_gestures   = groupby(pred_gestures);
    score = levenshtein(char(actual_gestures+96), char(pred_gestures+96));
    fprintf('\nScored: %d',score)
    
    current_scores(ii) = score;
    end
    scores(:,num_particles) = [mean(current_scores); std(current_scores)]
    
    end
    scores
%}
    
    %% Particle Filtering
    fprintf('...Start particle filtering\n');
    score_list = [];
    num_particles = 50;
    
    for ii = 1:length(labels_test)
        [X_est, Z_est] = particle_filter_sam(num_particles*20, model_est, seqs_test{ii}, trans_est);
    
        %%% Testing Labels
        groupby = @(labels) ...
            [labels(1) labels(find((labels(1:end-1)-labels(2:end))~=0)+1)];
    
        actual_gestures = groupby(labels_test{ii});
        [~,pred_gestures] = max(Z_est);
        pred_gestures   = groupby(pred_gestures);
        score = levenshtein(char(actual_gestures+96), char(pred_gestures+96));
        fprintf('\nScored: %d',score)
        score_list(end+1) = score;
    end
    fprintf('\nMean Score: %d',mean(score_list))
