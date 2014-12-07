function [seq, data] = load_bees(dname, I_seq)
% LOAD_BEES - Load the "dancing honeybees" dataset.  
%
% Inputs:
%   dname: Directory containing bees data (string)
%   I_seq: Sequence indices to load.
%
% Outputs:
%    seq: struct containing data structures representing all sequences.
%         seq{i} represents the i-th sequence. seq{i} contains 5 fields:
%         [seq{i}.x, seq{i}.y, seq{i}.sin, and seq{i}.cos] is the pose
%         information of the bee, as outlined in the handout. seq{i}.label
%         is a discrete label {1,2,3} indicating the state the bee was in
%         at that time step in. Each field contains T elements, where T is
%         the length of the training sequence.
%   data: a 3x1 cell array, where each entry corresponds to pose
%         information for bees in the discrete states 1,2,3, respectively. 
%         numel(data{k}) is the number of maximally contiguous sequences a
%         bee displayed in the data. data{k}{i} is a 4 x T_{k,i} sequence of
%         poses for the k-th state and the i-th sequence. T_{k,i} is the
%         length of this particular sequence. 
%
% Brown CS242

  names = {'turn_left', 'turn_right', 'waggle'};
  num_states = numel( names );

  sequences = [1,2,3];
  data = cell([num_states, 1]);
  seq = cell([numel(I_seq), 1]);  
  
  for ind=1:numel(I_seq)
      s_i = I_seq(ind);
    s = sequences(s_i);
    
    
    % load data
    fname = sprintf('%s/bee%d.mat', dname, s);
    load(fname, 'x', 'y', 'theta', 'label');
    
    Nscans = numel(x);
    
    % store sequence
    [ seq{ind}.x, seq{ind}.y, seq{ind}.sin, seq{ind}.cos, seq{ind}.label ] = ...
      deal(x, y, sin(theta), cos(theta), label);
    label_int = zeros(size(seq{ind}.label));
    for k=1:num_states
      label_int( strcmp(seq{ind}.label,names{k}) ) = k;
    end
    seq{ind}.label = label_int;
    
    % format data sequences
    for k = 1:num_states

      % init
%       data{k} = {};
      new_seq = true;

      % loop over scans
      for i=1:Nscans

        % check state
        if ~strcmp( label{i}, names{k} )
          new_seq = true;
          continue;
        end

        % reset sequence iterator
        if new_seq
          i_seq = 1;
        end

        % in state k, add
        thisZ = [ x(i); y(i); sin(theta(i)); cos(theta(i)) ];
        if new_seq
          data{k}{end+1}(:,i_seq) = thisZ;      
        else
          data{k}{end}(:,i_seq) = thisZ;
        end            

        i_seq = i_seq+1;
        new_seq = false;
      end        
    end
  end  
end
