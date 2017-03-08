function [network, err] = train_par_tsne(train_X, layers, max_iter_bp, perplexity)
%TRAIN_PAR_TSNE Trains a parametric t-SNE embedding
%
%   [network, err] = train_par_tsne(train_X, train_labels, test_X,
%   test_labels, layers, training)
%
% Trains up a parametric t-SNE embedding with the structure that is
% specified in layers. The used training technique is specified in
% training. Possible values are 'CD1' and 'PCD' (default = 'CD1').
%
%
% (C) Laurens van der Maaten
% Maastricht University, 2008

% Pretrain the network
origX = train_X;
no_layers = length(layers);
network = cell(1, no_layers);
for i=1:no_layers
    
    % Print progress
    disp(['Training layer ', num2str(i), ' (size ', ...
        num2str(size(train_X, 2)), ' -> ', ...
        num2str(layers(i)), ')...']);
    
    if i ~= no_layers
        % Train the layers using CD-1
        network{i} = train_rbm(train_X, layers(i), 0.01, 30);
        
        % Transform data using learned weights
        train_X = 1 ./ (1 + exp(-(bsxfun(@plus, train_X * network{i}.W, ...
            network{i}.bias_upW))));
    else
        % Train layer using linear hidden units
        network{i} = train_lin_rbm(train_X, layers(i), 0.0001, 100);
    end
end

% Perform backpropagation of the network using t-SNE gradient
[network, err] = tsne_backprop(network(1:no_layers), origX, ...
    max_iter_bp, perplexity, 1);
