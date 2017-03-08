% Generates class balanced train/validation/test sets
%
% This file is part of parametric t-SNE applied to 3D-DESI imaging.
% 
%     3D_DESI is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     3D_DESI is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with 3D_DESI.  If not, see <http://www.gnu.org/licenses/>.
%
% Author: Paolo Inglese, Imperial College London, 2016

function [xtrain, xval, xtest] = makesets(x, y, numtrain, numval, randomize)

if(nargin==4)
    randomize = true;
end

unique_classes = unique(y);
num_classes = length(unique_classes);

xtrain = struct('samples', [], 'idx', [], 'labels', []);
xval   = struct('samples', [], 'idx', [], 'labels', []);
xtest  = struct('samples', [], 'idx', [], 'labels', []);

% define balanced training set
yidx = cell(num_classes,1);
for i = 1:num_classes
    yidx{i} = find(y==unique_classes(i));
end

num_train_per_class = repmat(round(numtrain / num_classes), num_classes, 1);
% if num. train samples is not a multiple of the number of classes, then
% add the remaining samples to the class 1
if(num_train_per_class(1) * num_classes ~= numtrain)
    num_train_per_class(1) = num_train_per_class(1) + numtrain - ...
        num_train_per_class(1)*num_classes;
end

for i = 1:num_classes
    idx_tmp        = yidx{i}(randsample(length(yidx{i}), ...
        num_train_per_class(i)));
    xtrain.idx     = [xtrain.idx; idx_tmp];
    xtrain.samples = [xtrain.samples; x(idx_tmp, :)];
end
xtrain.labels = y(xtrain.idx);

% remove the training indices from the selection
for i = 1:num_classes
    yidx{i}(ismember(yidx{i}, xtrain.idx)) = [];
end

% define balanced validation set
num_val_per_class = repmat(round(numval / num_classes), num_classes, 1);
% if num. val. samples is not a multiple of the number of classes, then
% add the remaining samples to the class 1
if(num_val_per_class(1) * num_classes ~= numval)
    num_val_per_class(1) = num_val_per_class(1) + numval - ...
        num_val_per_class(1) * num_classes;
end

for i = 1:num_classes
    if(length(yidx{i})<num_val_per_class(i))
        idx_tmp = yidx{i};
    else
        idx_tmp      = yidx{i}(randsample(length(yidx{i}), ...
            num_val_per_class(i)));
    end
    xval.idx     = [xval.idx; idx_tmp];
    xval.samples = [xval.samples; x(idx_tmp, :)];
end
xval.labels = y(xval.idx);

% remove the training indices from the selection
for i = 1:num_classes
    yidx{i}(ismember(yidx{i}, xval.idx)) = [];
end

xtest.idx = cell2mat(yidx);
xtest.samples = x(xtest.idx,:);
xtest.labels = y(xtest.idx);

% shuffle
if(randomize)
    
    rand_idx = randperm(length(xtrain.idx));
    xtrain.idx = xtrain.idx(rand_idx);
    xtrain.samples = xtrain.samples(rand_idx, :);
    xtrain.labels = xtrain.labels(rand_idx);
    
    rand_idx = randperm(length(xval.idx));
    xval.idx = xval.idx(rand_idx);
    xval.samples = xval.samples(rand_idx, :);
    xval.labels = xval.labels(rand_idx);
    
    rand_idx = randperm(length(xtest.idx));
    xtest.idx = xtest.idx(rand_idx);
    xtest.samples = xtest.samples(rand_idx, :);
    xtest.labels = xtest.labels(rand_idx);
    
end

end
