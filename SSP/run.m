clear variables;
num_states = 21;

    %% Get data
    inds = [2:3];
    % split into train/test
    [labels_train, data_train] = load_data('./data', inds);
    [labels_test, ~] = load_data('./data', 1);
    labels_test = labels_test{1};
    fprintf('done getting data')
    
    
    %% Run EM learning
    
    conv_tol = 1e-4;
    max_iters = 100;
    [model_init, x0_init, P0_init ] = init_model(num_states,size(data_train{1}{1},1),size(data_train{1}{1},1)); % "7*<num_joints>" depends on how many joints we use
    model_est = cell([num_states,1]);
    ll_iter = cell([num_states,1]);
    
    parfor kk = 1:num_states
        data_train{kk} = cellfun(@(x)x',data_train{kk},'UniformOutput',false);
        [ model_est{kk}, ll_iter{kk} ] = em_lds_general(model_init{kk}, x0_init, P0_init, data_train{kk}, max_iters, conv_tol);
    end

    fig = figure(); hold on
    for kk=2:num_states
        subplot(5,4,kk-1)
        plot( ll_iter{kk}, '-b', 'LineWidth', 3 );
        axis([1 numel(ll_iter{kk}) ll_iter{kk}(1)-10 ll_iter{kk}(end)+1000])
    end
    saveas(fig, 'logLike.pdf');

    
    %xlabel('Iteration #');
    %ylabel('Log-Likelihood');
    %title(sprintf('Training Log-Likelihood SEQ:(%d,%d)',inds(1),inds(2)));
