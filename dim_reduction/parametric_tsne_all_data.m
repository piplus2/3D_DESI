% Main routine for the parametric t-SNE dimensionality reduction and
% comparison with PCA. It requires a data structure called 'data'
% containing a member 'data.Y' representing the MS intensity matrix
% (pixels x ions) and a member 'data.supervised.predLabels' containing
% a vector (pixels x 1) with the predicted tissue labels.
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

clear, clc

% Add parametric t-SNE folder to the path
addpath('./Parametric t-SNE/');

% Make the folders for figures and results
mkdir('../results_/all_data/pca/');
mkdir('../figures_/all_data/');

% Load the segmented data
load('../data_/Colorectal_3D_segmented.mat');

% Number of repetitions
NUM_REP = 5;

% Number of scaling methods tried for PCA
NUM_SCALING_METHODS = 6;

% Set sizes
SIZE_TRAIN = 30000;
SIZE_VAL   = 10000;

% Trustworthiness
T = zeros(NUM_REP,1);

% k-NN accuracies
MAX_K = 20;
knn_accuracy_pca = zeros(MAX_K, NUM_REP, NUM_SCALING_METHODS);
knn_accuracy_tsne = zeros(MAX_K, NUM_REP);

% Perform the analysis NUM_REP times
for rep = 1:NUM_REP
    
    fprintf(2,'rep %d/%d\n', rep, NUM_REP);
    
    % Create datasets for training/test
    clc
    
    [xtrain, xval, xtest] = makesets(data.Y, data.supervised.predLabels, ...
        SIZE_TRAIN, SIZE_VAL, true);
    
    fprintf('num train = %d\n', size(xtrain.samples,1));
    fprintf('num valid. = %d\n', size(xval.samples,1));
    fprintf('num test = %d\n', size(xtest.samples,1));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate the PCA scores
    % Try different scaling methods, from
    % van den Berg, Robert A., et al. "Centering, scaling,
    % and transformations: improving the biological information content of 
    % metabolomics data." BMC genomics 7.1 (2006): 1.
    scaling_methods = {'center', 'autoscaling', 'range', 'pareto', ...
        'vast', 'level'};
    pca_scores = zeros(size(data.Y, 1), 2, length(scaling_methods));
    for im = 1:length(scaling_methods)
        % Determine the scaling parameters from the training set and apply
        % the scaling on the entire dataset
        Xscaled = scaleIntensity(data.Y, xtrain.idx, scaling_methods{im});
        % Centered = false, we are using already scaled data
        p = pca(Xscaled(xtrain.idx, :), 'numcomponents', 2, ...
            'Centered', false);
        % Calculate the PCA scores
        pca_scores(:,:,im) = Xscaled * p;
    end
    % Plot the PCA scores obtained with the different scaling methods
    figure();
    for im = 1:length(scaling_methods)
        subplot(2,3,im), gscatter(squeeze(pca_scores(:,1,im)), ...
            squeeze(pca_scores(:,2,im)), data.supervised.predLabels);
        title(scaling_methods{im});
        box on
        axis image
    end
    pcaFile = ['../results_/all_data/pca/scores_repetition_', ...
        int2str(rep), '.mat'];
    save(pcaFile, 'pca_scores');
    savefig(['../figures_/all_data/pca_repetition_', int2str(rep), ...
        '.fig']);
    close;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parametric t-SNE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Scale train-validation and test (THIS IS NECESSARY SINCE RBM REQUIRES
    % BERNOULLI-DISTRIBUTED INPUTS)
    disp('...applying MinMax scaling');
    m = min(xtrain.samples, [], 2); M = max(xtrain.samples, [], 2);
    xtrain.samples = bsxfun(@rdivide, bsxfun(@minus, xtrain.samples, m), M-m);
    m = min(xval.samples, [], 2); M = max(xval.samples, [], 2);
    xval.samples = bsxfun(@rdivide, bsxfun(@minus, xval.samples, m), M-m);
    m = min(xtest.samples, [], 2); M = max(xtest.samples, [], 2);
    xtest.samples = bsxfun(@rdivide, bsxfun(@minus, xtest.samples, m), M-m);
    clear m M
    
    % Define the topology of parametric t-SNE network
    max_iter = 500;
    perplexity = 30;
    layers = [250 250 1000 2];
    
    % Train the parametric t-SNE network
    [network, err] = train_par_tsne(xtrain.samples, layers, max_iter, ...
        perplexity);
    
    % Map the training/test data in the low-dimensional manifold
    % Construct training and test embeddings
    mapped_train_X = run_data_through_network(network, xtrain.samples);
    % In this case we test on the validation set because the test set is
    % too large to calculate the trustworthiness
    mapped_val_X = run_data_through_network(network, xval.samples);
    % The test set is used for k-NN
    mapped_test_X = run_data_through_network(network, xtest.samples);
    
    % Calculate the trustworthiness on the val/test set (by default it uses
    % the validation set since it can require a lot of memory for the test
    % set)
    fprintf('...calculating the thrustworthiness on validation data\n');
    T(rep) = trustworthiness(xval.samples, mapped_val_X, 12);
    fprintf('Trustworthiness: %f\n', T(rep));
    
    % Perform k-NN on PCA scores
    fprintf('...performing k-NN classification\n');
    for im = 1:length(scaling_methods)
        for k = 1:MAX_K
            mdl = fitcknn(squeeze(pca_scores(xtrain.idx,:,im)), ...
                xtrain.labels, 'numneighbors', k);
            ypred_pca = predict(mdl, squeeze(pca_scores(xtest.idx,:,im)));
            knn_accuracy_pca(k,im,rep) = nnz(ypred_pca == xtest.labels) ...
                / length(xtest.labels);
        end
    end
    clear ypred_pca
    % Plot the results
    figure;
    plot(knn_accuracy_pca(:,:,rep));
    legend(scaling_methods, 'Location', 'SE');
    savefig(['../figures_/all_data/knn_accuracy_pca_repetition_', ...
        int2str(rep), '.fig']);
    close;
    
    save(['../results_/all_data/knn_accuracy_pca_repetition_', ...
        int2str(rep), '.mat'], 'knn_accuracy_pca');
    
    % Perform k-NN on mapped data
    for k = 1:MAX_K
        mdl = fitcknn(mapped_train_X, xtrain.labels, 'numneighbors', k);
        ypred_enc = predict(mdl, mapped_test_X);
        knn_accuracy_tsne(k,rep) = nnz(ypred_enc == xtest.labels) ...
            / length(xtest.labels);
    end
    clear ypred_enc
    save(['../results_/all_data/knn_accuracy_pca_repetition_', ...
        int2str(rep), '.mat'], 'knn_accuracy_tsne');
    
    % Map all the data in the low-dimensional manifold
    X = data.Y;
    m = min(X, [], 2);
    M = max(X, [], 2);
    X = bsxfun(@rdivide, bsxfun(@minus, X, m), M-m); % First MinMax scaling
    clear m M
    
    mapped_X = run_data_through_network(network, X);
    
    % Save the network and the mapped data
    save(['../results_/all_data/par_tsne_model_repetition_', ...
        int2str(rep), '.mat'], 'network');
    save(['../results_/all_data/mapped_data_repetition_', ...
        int2str(rep), '.mat'], 'mapped_X');
        
    clear X mapped_X
    
end
