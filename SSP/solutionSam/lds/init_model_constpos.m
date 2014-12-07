function [ model, x0, P0 ] = init_model_constpos( D, M, sig_x, sig_y )
% INIT_MODEL_CONSTPOS - Wrapper function to constant position model
%
% INPUTS:
%   D - State dimension
%   M - Measurement dimension
%   sig_x - STDEV of latent state dimensions
%   sig_y - STDEV of observation dimensions
%
% Brown CS242
  
  L = D-M;

  model.A = eye(D);
  if L>0
    model.C = [ eye(D), zeros([M,D]) ];
  else
    model.C = eye(M);
  end
  model.Q = sig_x^2 * eye(D);
  model.R = sig_y^2 * eye(M);  
  x0 = zeros([D,1]);
  P0 = eye(D);
  
end
