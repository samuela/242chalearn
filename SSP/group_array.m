function [seq] = group_array( labels )
% Given a list of labels
%   Return the list of contiguous groupings of the labels
  seq = [];
  if length(labels)>1
    seq = [labels(1)];
    for ii=1:numel(labels)
        if seq(end)~=labels(ii)
            seq(end+1)=labels(ii);
        end
    end
  end

end

