function [labels, data, seqs] = load_data(dname, I_seq)
% Load the data
%
% Inputs:
%   dname: Directory containing bees data (string)
%   I_seq: Sequence indices to load.
%
% Outputs:
% labels: struct containing label data for all sequences.
%         labels{i} represents the i-th sequence. labels{i} is a 1 by T array
%         of discrete labels from 1 to 21 where T is
%         the length of the training sequence.
%   data: a 21x1 cell array, where each entry corresponds to pose
%         information for gestures 1,2,...,21, respectively.
%         numel(data{k}) is the number of maximally contiguous sequences a
%         of a gesture. data{k}{i} is a 140 x T_{k,i} sequence of
%         poses for the k-th gesture and the i-th sequence. T_{k,i} is the
%         length of this particular sequence.
%   seqs: struct containing joint data for all sequences. seqs{i}
%         represents the i-th sequence. seqs{i} is a numel(joints_to_use) by T array
%         of values for the joints used in each sequence where T is the
%         length of the training sequence.

joints_to_use = 1:20; 

all_labels = char('*NONE*', 'vattene', 'vieniqui', 'perfetto', 'furbo', 'cheduepalle', 'chevuoi', 'daccordo', 'seipazzo', ...
             'combinato','freganiente','ok','cosatifarei','basta','prendere','noncenepiu','fame',...
             'tantotempo','buonissimo','messidaccordo','sonostufo');

[num_states, ~] = size(all_labels);

sequences = char('00710','00050','00279');
%sequences = char('00403', '00058', '00059', '00060', '00061', '00062', '00063', '00064', '00065', '00066', '00067',...
%    '00068', '00069', '00070', '00071', '00072', '00073', '00074', '00075');
data = cell([num_states, 1]);
labels = cell([numel(I_seq), 1]);
seqs = cell([numel(I_seq), 1]);

for ind=1:numel(I_seq)
    s_i = I_seq(ind);
    s = sequences(s_i,:);
    
    % load data
    fname = sprintf('%s/Sample%s_data.mat', dname, s);
    load(fname, 'Video');
    
    none_index = 1;
    labels{ind} = zeros(1,Video.Labels(numel(Video.Labels)).End);
    seqs{ind} = zeros(7*numel(joints_to_use),Video.Labels(numel(Video.Labels)).End);
    
    %go through each label in order of occurence
    for i=1:numel(Video.Labels)
       info = Video.Labels(i);
       k = strmatch(info.Name, all_labels, 'exact'); %which gesture is it?
       num_seq_k = numel(data{k});
       data{k}{num_seq_k+1} = [];
       %fill in data{k}{num_seq+1}
       for j=info.Begin:info.End
           positions = Video.Frames(j).Skeleton.WorldPosition;
           rotations = Video.Frames(j).Skeleton.WorldRotation;
           % *perhaps fewer joints could be needed for SSP
           positions = positions(joints_to_use,:); 
           rotations = rotations(joints_to_use,:);
           positions = reshape(positions, 1, numel(positions));
           rotations = reshape(rotations, 1, numel(rotations));
           data{k}{num_seq_k+1}(:,end+1) = [positions,rotations];
           seqs{ind}(:, j) = [positions,rotations];
       end
       labels{ind}(info.Begin:info.End) = k;
       %fill in data{1}{num_seq+1}, the NONE sequence
       if none_index < info.Begin-1
           num_seq_1 = numel(data{1});
           data{1}{num_seq_1+1} = [];
           for j=none_index:info.Begin-1
               positions = Video.Frames(j).Skeleton.WorldPosition;
               rotations = Video.Frames(j).Skeleton.WorldRotation;
               % *perhaps fewer joints could be needed for SSP
               positions = positions(joints_to_use,:); 
               rotations = rotations(joints_to_use,:);
               positions = reshape(positions, 1, numel(positions));
               rotations = reshape(rotations, 1, numel(rotations));
               data{1}{num_seq_1+1}(:,end+1) = [positions,rotations];
               seqs{ind}(:, j) = [positions,rotations];
           end
           labels{ind}(none_index:info.Begin-1) = 1;
       end
       none_index = info.End+1;
    end
  
end


end
