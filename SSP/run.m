clear variables;

    %% Get data
    inds = [2:3];
    % split into train/test
    [labels_train, data_train] = load_data('./data', inds);
    [labels_test, ~] = load_data('./data', 1);
    labels_test = labels_test{1};
    'done getting data'
    
    %% Run EM learning
    
    conv_tol = 1e-4;
    max_iters = 100;
    [model_init, x0_init, P0_init ] = init_model(21,140,140);

    [ model_est, x0_est, P0_est, ll_iter ] = ...
      em_slds(model_init, x0_init, P0_init, { labels_test }, data_train, max_iters, conv_tol);
    
    fig = figure();
    plot( ll_iter, '-b', 'LineWidth', 3 );
    xlabel('Iteration #');
    ylabel('Log-Likelihood');
    title(sprintf('Training Log-Likelihood SEQ:(%d,%d)',inds(1),inds(2)));
    saveas(fig, 'logLike.pdf');
