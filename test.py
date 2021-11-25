import scipy.io as sio
import time
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

import os
from constants import *

# Include standard modules
import getopt, sys

# Get full command-line arguments
full_cmd_arguments = sys.argv

argument_list = full_cmd_arguments[1:]

print(argument_list)

short_options = "d:c:"
long_options = ["dataset=", "clusters=", "square_dist"]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-d", "--dataset"):
        DATASET_NAME = current_value
    elif current_argument in ("-c", "--clusters"):
        N_CLUSTERS = int(current_value)
        TEST = False
    elif current_argument in ("--square_dist"):    
        SQAURE_DIST = True


def normalize_adj(adj, type_='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type_ == 'sym':
        adj = sp.coo_matrix(adj)
        # sum over the rows of the matrix
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # set inf or -inf values to 0
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # get D^{-1/2} diagonal matrix
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # return D^{-1/2} A D^{-1/2}
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type_ == 'rw':
        # sum over the rows of the matrix
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        # set inf or -inf values to 0
        d_inv[np.isinf(d_inv)] = 0.
        # get D^{-1}
        d_mat_inv = sp.diags(d_inv)
        # return D^{-1} A
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type_='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        # return D^{-1/2} (A+I) D^{-1/2}
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type_=type_)
    return adj_normalized

# return one hot representation of the given vector
def to_onehot(prelabel):
    # number of different values
    k = len(np.unique(prelabel))
    # matrix of zeros with shape prelabel.rows, k
    label = np.zeros([prelabel.shape[0], k])
    # a 1 for each row in corrispondence with the value in prelabel
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def square_dist(prelabel, feature):
    # pass features to dense
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)

    # get the one hot encodig of the labels
    # where on each row there is a different label
    onehot = to_onehot(prelabel)

    # m = num of labels
    # n = num of nodes
    m, n = onehot.shape
    # count the number of elements assigned to each label
    count = onehot.sum(1).reshape(m, 1)
    # put to 1 where is zero in order to avoid division by 0
    count[count==0] = 1
    # in each row we have the avg of the nodes feature associated to the same label
    mean = onehot.dot(feature)/count
    
    # sum over the labels of (avg vector of squared features)
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    # distance
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    # distance among the different labels
    intra_dist = pdist2.trace()
    intra_dist /= m
    
    # distance from each label to the other ones
    inter_dist = pdist2.sum() - intra_dist
    inter_dist /= m * (m - 1)

    return intra_dist

def dist(prelabel, feature):
    # num of different labels
    k = len(np.unique(prelabel))
    intra_dist = 0
    # for each label
    for i in range(k):
        # get features of nodes associated to label i
        Data_i = feature[np.where(prelabel == i)]
        # get the euclidean distance among all the pair of nodes associated to label i
        Dis = euclidean_distances(Data_i, Data_i)
        # get number of nodes associated to label i
        n_i = Data_i.shape[0]
        
        if n_i == 0 or n_i == 1:
            intra_dist = intra_dist
        else:
            # if more than one node
            # (sum of all the pair distances) / (|C|(|C|-1))
            # where C is the set of nodes associated to label i 
            intra_dist = intra_dist + 1/k * 1/(n_i * (n_i - 1)) * sum(sum(Dis))

    return intra_dist

# sparse matrix to (coords of the edges, values of the edges, shape of the matrix)
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# get adj_train, train a and test edges
def get_test_edges(adj, test_size=0.1, train_size=0.15):
    adj_ = sp.triu(adj)
    coords, _, shape = sparse_to_tuple(adj_)
    
    all_edges = coords
    
    # define the number of edges for train and test
    num_train = int(train_size*all_edges.shape[0])
    num_test = int(test_size*all_edges.shape[0])

    # shuffle the edges
    np.random.shuffle(all_edges)

    # get the first num_test edges (after shuffling)
    # as the test_edges (the positive ones)
    test_edges = all_edges[:num_test]
    # get the first num_train after the first num_test edges (after shuffling)
    # as the train_edges (the positive ones)
    train_edges = all_edges[num_test:num_test+num_train]
    res_edges = all_edges[num_test+num_train:]
    

    n_nodes = adj_.shape[0]
    # with this method we keed the proportions the same in res, train and test
    #n_false_train_edges = int(((n_nodes**2 - len(res_edges))/len(res_edges))*len(train_edges))
    #n_false_test_edges = int(((n_nodes**2 - len(res_edges))/len(res_edges))*len(test_edges))
    
    n_false_train_edges = len(train_edges)
    n_false_test_edges = len(test_edges)
    
    print(f"train_edges: {len(train_edges)}")
    print(f"res_edges: {len(res_edges)}")
    print(f"train_false: {n_false_train_edges}, test_false: {n_false_test_edges}")
    print(f"false: {n_false_train_edges + n_false_test_edges}")
    print(f"total_false: {n_nodes**2 - len(res_edges) - len(train_edges) - len(test_edges)}")
    
    # turn the remaning edges into a sparse matrix
    adj_train = sp.csr_matrix((np.ones(res_edges.shape[0]), (res_edges[:, 0], res_edges[:, 1])), shape=adj.shape)

    return adj_train, res_edges, train_edges, test_edges 


def save_split(res_edges, train_edges, test_edges):
    with open(os.sep.join([OUTPUT_DIR,"res_edges.csv"]), "w") as fout:
        for edge in res_edges:
            fout.write(f"{edge[0]},{edge[1]}\n")
    with open(os.sep.join([OUTPUT_DIR,"train_edges.csv"]), "w") as fout:
        for edge in train_edges:
            fout.write(f"{edge[0]},{edge[1]}\n")
    with open(os.sep.join([OUTPUT_DIR,"test_edges.csv"]), "w") as fout:
        for edge in test_edges:
            fout.write(f"{edge[0]},{edge[1]}\n")

if __name__ == '__main__':
    # load data
    data = sio.loadmat(os.sep.join([DATASET_DIR,f'{DATASET_NAME}.mat']))
    # get the features
    feature = data['fea']
    # if the matrix are sparse, turn them to dense
    if sp.issparse(feature):
        feature = feature.todense()
    # get the adj data
    adj = data['W']
    
    # split the edges
    adj_train, res_edges, train_edges, test_edges = get_test_edges(adj, 0.2,0.2)
    # save the split
    save_split(res_edges, train_edges, test_edges)
    # get_test_edges returns a triu matrix, we want the complete one
    adj = adj_train + adj_train.T
    # get gnd labeling
    gnd = data['gnd']
    gnd = gnd.T
    # pass it to zero base
    gnd = gnd - 1
    # remove a dimension and obtain a 1-D vector
    gnd = gnd[0, :]

    if TEST:
        # get the number of correct clusters
        k = len(np.unique(gnd))
    else:
        k = N_CLUSTERS

    # pass matrix to coo format
    adj = sp.coo_matrix(adj)
    
    # initialize the list of intra cluster distances
    intra_list = [float('inf')]
    
    acc_list, nmi_list, f1_list = [], [], []
    stdacc_list, stdnmi_list, stdf1_list = [], [], []
    
    max_iter, rep =60, 10
    
    t = time.time()
    adj_normalized = preprocess_adj(adj)
    adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2
    total_dist = []

    tt = 0
    while 1:
        tt = tt + 1
        power = tt
        # initialize the intradistance to a zero vector
        intraD = np.zeros(rep)
        ac = np.zeros(rep)
        nm = np.zeros(rep)
        f1 = np.zeros(rep)

        # compute A*X
        # where X is the matrix of features at time zero
        feature = adj_normalized.dot(feature)

        # u -> left sing vectors as columns
        # s -> singular values
        # v -> right singular values
        u, s, v = sp.linalg.svds(feature, k=k, which='LM')
    
        predict_labels = None
        for i in range(rep):
            # run kmeans on the current features
            kmeans = KMeans(n_clusters=k).fit(u)
            # get the predicted labels
            predict_labels = kmeans.predict(u)
            
            # measure the intra distance of the clusters
            if SQAURE_DIST:
                intraD[i] = square_dist(predict_labels, feature)
            else:
                intraD[i] = dist(predict_labels, feature)
            
            if TEST:
                # measure the accuracy, F1-score and nmi
                cm = clustering_metrics(gnd, predict_labels)
                ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()
        
        # save the scores 
        intra_list.append(np.mean(intraD))
        
        if TEST:
            acc_list.append(np.mean(ac))
            stdacc_list.append(np.std(ac))
            nmi_list.append(np.mean(nm))
            stdnmi_list.append(np.std(nm))
            f1_list.append(np.mean(f1))
            stdf1_list.append(np.std(f1))
            
            print(f'power: {power}',
              f'acc_mean: {acc_list[-1]}',
              f'acc_std: {stdacc_list[-1]}',
              f'nmi_mean: {nmi_list[-1]}',
              f'nmi_std: {stdnmi_list[-1]}',
              f'f1_mean: {f1_list[-1]}',
              f'f1_std: {stdf1_list[-1]}')
        
        print(f'intra_dist: {intra_list[-1]}')
              
        # if the intra distance is increased wrt the precedent run stop. 
        if intra_list[tt] > intra_list[tt - 1] or tt > max_iter:
            print(f'bestpower: {tt-1}')
            t = time.time() - t
            print(t)
            
            # save labels
            with open(os.sep.join([OUTPUT_DIR, f"labels_{DATASET_NAME}.csv"]), "w") as fout:
                for i in predict_labels:
                    fout.write(f"{i}\n")
            break





