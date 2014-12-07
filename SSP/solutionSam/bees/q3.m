clear variables;
DEBUG = 0;

seed = 0;
rng(seed);

% init stuff
D = 4;
M = 4;
num_states = 3;
num_seq = 3;
colors = {'.-r','.-g','.-b' ,'.-m'};

% figCount = 1;

%% MAIN LOOP
for i=1:num_seq
    inds = 1:3;
    inds(i) = [];
    inds
    % split into train/test
    [seq_train, data_train] = load_bees('.', inds);
    [seq_test, ~] = load_bees('.', i);
    seq_test = seq_test{1};
    
    %% 3a %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute ML estimates of the transition probabilities in the training set.
    % Compute using the data in seq_train.
    % YOUR CODE HERE!
    pi = zeros(num_states);
    for j=1:numel(seq_train)
      pi = pi + full(sparse(seq_train{j}.label(1:end-1), seq_train{j}.label(2:end), 1));
    end
    pi = bsxfun(@rdivide, pi, sum(pi, 2));
    
    %% 3d %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    conv_tol = 1e-4;
    max_iters = 100;
    [model_init, x0_init, P0_init] = init_model_bee();
    
    % YOUR EM CODE HERE! This should be an extension to your EM code from
    % Question 2.
    model = cell(num_states, 1);
    for k=1:num_states
      [model{k}, ll_iter] = em_lds_general(model_init{k}, x0_init, P0_init, ...
                                           data_train{k}, max_iters, conv_tol);

%       k
%       'A'
%       model{k}.A
%       'Q'
%       model{k}.Q
%       'C'
%       model{k}.C
%       'R'
%       model{k}.R
                                       
%       figure(figCount); figCount = figCount + 1;
      figure();
      plot( ll_iter, '-b', 'LineWidth', 3 );
      xlabel('Iteration #');
      ylabel('Log-Likelihood');
      title(sprintf('Training Log-Likelihood SEQ:(%d,%d)',inds(1),inds(2)));
      saveas(gcf, sprintf('q3d_split_%d_%d_k_%d.png', inds(1), inds(2), k));
    end

    %% 3e %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % make noisy test data
    T = numel(seq_test.x);
    noisy_seq_test = seq_test;
    vals = vectorize_seq(noisy_seq_test);
    for t=1:T
        val = vals(:,t);
        if rand() <= 0.9
            val = mvnrnd(val, 0.1*eye(M));
        else
            val = mvnrnd(zeros(M,1), 5*eye(M));
        end
        noisy_seq_test.x(t) = val(1);
        noisy_seq_test.y(t) = val(2);
        noisy_seq_test.sin(t) = val(3);
        noisy_seq_test.cos(t) = val(4);
    end

    noisy_poses = vectorize_seq(noisy_seq_test);
    clean_poses = vectorize_seq(seq_test);

    % visualize noisy data
    for st=1:2
%         figure(figCount); figCount=figCount+1;
        figure();
        if st == 1
          pose_use = noisy_poses; str = 'noisy poses';
        else
          pose_use = clean_poses; str = 'clean poses';
        end
        hold on;
        for d=1:4
            plot(1:size(pose_use,2),pose_use(d,:),colors{d});
        end
        hold off;
        xlabel('Time sequence');
        ylabel('Value');
        legend({'x','y','sin','cos'});
        title(['Visualization of the ',str]);
        ylim([-8,8]);
    end

    %% 3f %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T = numel(seq_test.x);
    num_particles = 1000;
    % Extend your code from Question 1c and use it here to analyze the
    % bee data. We need the posterior estimates of the state of the bee in
    % the test sequence, seq_test.
%     X_pf = zeros(4,T);
    % X_pf = ????
    [X_pf, Zpost] = particle_filter_noisy(num_particles, model, noisy_poses, pi);

    %% 3g %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     p_k = zeros(num_states, T);
    % Compute probability that each behaviour is active over time.
    p_k = Zpost;

    % plot discrete state probabilities
%     figure(figCount); figCount = figCount + 1;
    figure();
    hold on;
    for state=1:num_states
      plot( 1:T, p_k(state,:), colors{state}, 'LineWidth', 2 );
    end
    xlabel('Time');
    ylabel('Prob. of Discrete State');
    legend('Turn Left', 'Turn Right', 'Waggle');
    title(sprintf('Switching State Probability (Sequence %d)', i));
    hold off;
    saveas(gcf, sprintf('q3g_split_%d_%d_z_prob.png', inds(1), inds(2)));
    
    labels = seq_test.label;
%     figure(figCount); figCount = figCount + 1;
    figure();
    colors_no_line = {'.r','.g','.b'};
    hold on;
    for k=1:num_states
      ix = find(labels == k);
      plot(ix,ones(numel(ix),1),colors_no_line{k});
    end
    hold off;
    xlabel('Time');
    ylabel('Actual Discrete State');
    legend('Turn Left', 'Turn Right', 'Waggle');
    title(sprintf('Actual states for sequence %d', i));
    saveas(gcf, sprintf('q3g_split_%d_%d_z_actual.png', inds(1), inds(2)));
    
    % plot position
    % Positions earlier in the time sequence are darker red and smaller.
    % Position later in the sequence are lighter red and bigger. Lines
    % connect sequential position estimates.
    for st=1:2
        if st == 1
            x = X_pf(1,:);
            y = X_pf(2,:);
            str = 'Estimated';
        else
            x = seq_test.x;
            y = seq_test.y;
            str = 'Actual';
        end
%         figure(figCount); figCount=figCount+1;
        figure();
        hold on;
        for jj=1:numel(x)
            colour = [(jj-1)/numel(x),0,0];
            wd = 5*(jj-1)/numel(x)+3;
            plot(x(jj),y(jj),'x','Color',colour, 'Linewidth',wd);
        end
        plot(x,y,'-b');
        xlabel('X position');
        ylabel('Y position');
        title([str, ' 2D-position']);
        hold off;
        if st == 1
          saveas(gcf, sprintf('q3g_split_%d_%d_pos_pred.png', inds(1), inds(2)));
        else
          saveas(gcf, sprintf('q3g_split_%d_%d_pos_actual.png', inds(1), inds(2)));
        end
    end
    
    % plot angle
%     figure(figCount); figCount=figCount+1;
    figure();
    theta_pf = atan2( X_pf(3,:), X_pf(4,:) );
    theta_true = atan2( seq_test.sin, seq_test.cos );
    hold on;
    plot( 1:T, theta_pf, '--r' );
    plot( 1:T, theta_true, '-k', 'LineWidth', 3 );
    legend('Estimated Angle', 'True Angle');
    xlabel('Time');
    ylabel('Angle (radians)');
    title(sprintf('Estimated Angle (Sequence %d)', i));
    hold off;
    saveas(gcf, sprintf('q3g_split_%d_%d_angles.png', inds(1), inds(2)));

end
