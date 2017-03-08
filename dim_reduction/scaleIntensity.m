function [Xscaled, params] = scaleIntensity( X, train_idx, method )
%SCALEINTENSITY determine scaling method parameters from training data and
%apply the scaling to the entire dataset.
%
% Ref.
% van den Berg, Robert A., et al. "Centering, scaling,
% and transformations: improving the biological information content of 
% metabolomics data." BMC genomics 7.1 (2006): 1.
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

switch method
    case 'center'
        % X - mean(X)
        params.mean = mean(X(train_idx,:),1);
        Xscaled = bsxfun(@minus, X, params.mean);
    case 'autoscaling'
        % (X - mean(X)) / std(X)
        params.mean = mean(X(train_idx,:), 1);
        params.std = std(X(train_idx,:), [], 1);
        Xscaled = bsxfun(@rdivide, bsxfun(@minus, X, params.mean), params.std);
    case 'range'
        % (X - mean(X)) / (max(X) - min(X))
        params.mean = mean(X(train_idx,:), 1);
        params.range = range(X(train_idx,:), 1);
        Xscaled = bsxfun(@rdivide, bsxfun(@minus, X, params.mean), params.range);
    case 'pareto'
        % (X - mean(X)) / sqrt(std(X))
        params.mean = mean(X(train_idx,:), 1);
        params.square_std = square(std(X(train_idx,:), [], 1));
        Xscaled = bsxfun(@rdivide, bsxfun(@minus, X, params.mean), params.square_std);
    case 'vast'
        % (X - mean(X)) / std(X) * mean(X) / std(X)
        params.mean = mean(X(train_idx,:), 1);
        params.std = std(X(train_idx,:), [], 1);
        X1 = bsxfun(@rdivide, bsxfun(@minus, X, params.mean), params.std);
        X2 = bsxfun(@rdivide, params.mean, params.std);
        Xscaled = bsxfun(@times, X1, X2);
    case 'level'
        % (X - mean(X)) / mean(X)
        params.mean = mean(X(train_idx,:), 1);
        Xscaled = bsxfun(@rdivide, bsxfun(@minus, X, params.mean), params.mean);
    otherwise
        error('Scaling method not recognised');
end

end

