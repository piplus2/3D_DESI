# 3D_DESI

Matlab scripts for dimensionality reduction of DESI imaging dataset using the parametric t-SNE method.

The script trains a 391-250-250-1000-2 parametric t-SNE of a subset of the entire dataset (tumour + healthy + background)
and calculates the PCA with 5 scaling methods as a comparison.

A kNN model is trained on the low-dimensional data points to compare the performances of the parametric t-SNE method and
PCA.

Ref:
Inglese, P., McKenzie, J.S., Mroz, A., Kinross, J., Veselkov, K., Holmes, E., Takats, Z., Nicholson, J.K. and Glen, R.C., 2017. Deep learning and 3D-DESI imaging reveal the hidden metabolic heterogeneity of cancer. Chemical Science.

http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC03738K

The original MATLAB code for parametric t-SNE is available at:
https://lvdmaaten.github.io/tsne/code/ptsne.tar.gz
