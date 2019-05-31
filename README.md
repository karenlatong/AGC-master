# AGC-master
There are two ways for computing intra_cluster distance:
1) squared 2-norm distance: square_dist(predict_labels, feature)
2) 2-norm distance: dist(predict_labels, feature)
We recommend to use squared 2-norm distance, since squared 2-norm distance is usually more faster than 2-norm distance.

Usage
python test.py

Please kindly cite our paper if you use this code in your own work:
Xiaotong Zhang, Han Liu, Qimai Li and Xiao-Ming Wu, Attributed Graph Clustering via Adaptive Graph Convolution, IJCAI, 2019.

As the data features are nonnegative (the filtered features are also nonnegative), the similarity matrix W is the kernel matrix XX^T. Learning the eigenvectors of the kernel matrix XX^T is equivalent to computing the left singular vectors of X by SVD.

