clear variables;
close all;

seed = 0;
rng(seed);
figCount = 1;

%% 1a %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('track','states','data','model','x0','P0');
% Model is a data structure that contains fields: A,Q,R,C, which correspond
% to the parameters defined in Question 1 of the handout.
T = numel(data);

[Xf, Pf] = kalman_smoother(model, data', x0, P0);
close all;
plot_truth(states,data,figCount); figCount=figCount+1;
hold on;
E = shiftdim(2*sqrt( Pf(1,1,:) ), 1);
errorbar( 1:T, Xf(1,:), E, '.k' );
hold off;
xlim([0, T+1]);
legend('True State', 'Measurement', 'Kalman Filter');

clear Xf Pf
%% 1b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set initial conditions
% Todo: MODIFY 'P0_1d' and 'x0_1d' to the corresponding 1-d initial conditions
P0_1d = P0(1);
x0_1d = x0(1);    

model_1d = model;
% Todo: MODIFY 'model_1d' data structure to use the 1d case here

model_1d.A = 1;
model_1d.Q = 0.01/3;
model_1d.C = 1;

[Xf, Pf] = kalman_smoother(model_1d, data', x0_1d, P0_1d);
plot_truth(states, data, figCount); figCount=figCount+1;
hold on;
E = shiftdim(2*sqrt( Pf(1,1,:) ), 1);
errorbar( 1:T, Xf(1,:), E, '.k' );
hold off;
xlim([0, T+1]);
legend('True State', 'Measurement', 'Kalman Filter');

model_1d = model;
% Todo: MODIFY 'model_1d' data structure to use the 1d case here
model_1d.A = 1;
model_1d.Q = 10;
model_1d.C = 1;

[Xf, Pf] = kalman_smoother(model_1d, data', x0_1d, P0_1d);
plot_truth(states, data, figCount); figCount=figCount+1;
hold on;
E = shiftdim(2*sqrt( Pf(1,1,:) ), 1);
errorbar( 1:T, Xf(1,:), E, '.k' );
hold off;
xlim([0, T+1]);
legend('True State', 'Measurement', 'Kalman Filter');

clear x0_1d P0_1d model_1d

%% 1d %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numRuns = 3;
nParticles = 100;

plot_truth(states, data, figCount); figCount = figCount+1;
hold on;
E = shiftdim(2*sqrt( Pf(1,1,:) ), 1);
errorbar( 1:T, Xf(1,:), E, '.k' );
% run particle filter
for i=1:numRuns
    X_pf = particle_filter(nParticles, model, data', x0, P0);
    plot( 1:numel(data), X_pf(1,:), '.-c' );
    
end

xlim([0, T+1]);
h=legend('True State', 'Measurement', 'Kalman Filter', 'Particle Filter');
rect = [0.6, 0.25, .25, .25];
set(h, 'Position', rect)

clear X_pf
%% 1e %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numRuns = 3;
nParticles = 20;

plot_truth(states, data, figCount); figCount = figCount+1;
hold on;
E = shiftdim(2*sqrt( Pf(1,1,:) ), 1);
errorbar( 1:T, Xf(1,:), E, '.k' );
% run particle filter
for i=1:numRuns
    X_pf = particle_filter(nParticles, model, data', x0, P0);
    plot( 1:numel(data), X_pf(1,:), '.-c' );
    
end

xlim([0, T+1]);
h=legend('True State', 'Measurement', 'Kalman Filter', 'Particle Filter');
rect = [0.6, 0.25, .25, .25];
set(h, 'Position', rect)

clear X_pf

%% 1f %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nParticles = 100;

% corrupt data
to_flip= rand(numel(data),1) < 0.1;
noisy_vals = 40*randn(numel(data),1);
noisy_data = (1-to_flip).*data + to_flip.*noisy_vals;

% run Kalman smoother
[Xf, Pf] = kalman_smoother(model, noisy_data', x0, P0);

% run particle filter for the noisy observation model
% Todo: modify your implementation of particle_filter.m to take into
% account the outlier noise model

% place your posterior-mean estimates in X_pf. The code below is just a
% place holder for your code
X_pf = zeros(numel(x0),numel(data));
X_pf = particle_filter_noisy(nParticles, model, noisy_data', x0, P0);

% plot
plot_truth(states, noisy_data, figCount);figCount = figCount+1;
hold on;
E = shiftdim(2*sqrt( Pf(1,1,:) ), 1);
errorbar( 1:T, Xf(1,:), E, '.k' );
plot( 1:T, X_pf(1,:), '.-c' );
hold off;
xlim([0, T+1]);
legend('True State', 'Measurement', 'Kalman Filter', 'Particle Filter');
