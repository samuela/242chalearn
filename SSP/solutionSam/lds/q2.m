clear variables;
close all;

addpath('../');

seed = 0;
rng(seed);
figCount = 1;

%% 2a %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('track','states','data','model','x0','P0');
T = numel(data);

% run Kalman filter and smoother
[ Xf, Pf, Xs, Ps ] = kalman_smoother(model, data', x0, P0);

plot_truth(states, data, figCount); figCount = figCount+1;
hold on;
E = shiftdim(2*sqrt( Pf(1,1,:) ), 1);
errorbar( 1:T, Xf(1,:), E, '.k' );
E = shiftdim(2*sqrt( Ps(1,1,:) ), 1);
errorbar( 1:T, Xs(1,:), E, '.c' );
hold off;
xlim([0, T+1]);
legend('True State', 'Measurement', 'Kalman Filter', 'Kalman Smoother');
saveas(gcf, 'q2a.png');

%% 2d %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('spiral.mat','data');  

T = size(data,1);

D = 2;
M = 2;

% learn model
conv_tol = 1e-4;
max_iters = 100;
[ model_init, x0, P0 ] = init_model_constpos(D, M, 1, 1);

[ model_est, ll_iter ] = ...
    em_lds(model_init, x0, P0, data, max_iters, conv_tol);

model_est

figure(figCount); figCount = figCount + 1;
plot( 1:numel(ll_iter), ll_iter, '.-b' );
xlabel('Iteration #');
ylabel('Log-Likelihood Bound');
saveas(gcf, 'q2d_ll.png');

[ Xf, Pf, Xs, Ps ] = kalman_smoother(model_est, data', x0, P0);  
plot_truth2D( data', figCount); figCount = figCount+1;
hold on;
plot( Xf(1,:), Xf(2,:), '.-k' );
plot( Xs(1,:), Xs(2,:), '.-c' );
hold off;
legend('Observations', 'Kalman Filter', 'Kalman Smoother', 'Location', 'Best');
saveas(gcf, 'q2d_pos.png');
