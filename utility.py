import torch
import numpy as np
import scipy.io as scio
import os
import random
from tqdm import tqdm
from Deep_walks import deep_walk
import networkx as nx
import dgl
import argparse
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist




def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str)
    parser.add_argument('--seed', default=2023, type=int)

    # "subgraph VS subgraph"
    parser.add_argument('--psi', default=2, type=int)
    parser.add_argument('--h', default=3, type=int, help='the degree of subgraph')
    parser.add_argument('--lamda', default=0.0625, type=float)

    # "node VS node" "node VS subgraph"
    #parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)#100
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=150)#256
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--gama', type=float, default=1)
    parser.add_argument('--Sinkhorn_iter_times', type=int, default=5)
    parser.add_argument('--Sinkhorn_lamb', type=int, default=20)
    parser.add_argument('--topo_t', type=int, default=10, help='temperature for sigmoid in topology dist')
    parser.add_argument('--temperature', type=float, default=3, help='temperature for fx')
    parser.add_argument('--rectified', type=bool, help='use rectified cost matrix', default=True)
    parser.add_argument('--have_neg', type=bool, help='anomaly score and LOSS contain negtive pairs OT', default=True)
    parser.add_argument('--neg_top_k', type=float, help='top max k of OT to select negtive pairs', default=20)

    parser.add_argument('--K_1', type=int, help='view 1')
    parser.add_argument('--K_2', type=int, help='view 2')
    parser.add_argument('--restart_prob_1', type=float, help='RWR restart probability on view 1', default=0.9)
    parser.add_argument('--restart_prob_2', type=float, help='RWR restart probability on view 2', default=0.3)
    parser.add_argument('--subgraph_mode', type=str, default='1+2')#1+2

   
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight for inter-loss in total loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for combining two types of anomaly scores')

    args = parser.parse_args()

    return args

def set_seed(args):
    """
    :param args:
    :return:
    """

    torch.manual_seed(args.seed)       # Set PyTorch CPU seed
    torch.cuda.manual_seed_all(args.seed)  # Set PyTorch GPU seed (for all GPUs)
    np.random.seed(args.seed)          # Set NumPy seed
    random.seed(args.seed)             # Set Python random seed
    dgl.random.seed(args.seed)         # Set DGL random seed

    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_graph(dataset, train_rate=0.3, val_rate=0.1):
    #dir = 'dataset'
    #graph = scio.loadmat(os.path.join(dir, filename))
    """Load .mat dataset."""
    graph = scio.loadmat(os.path.join("..", "datasets", f"{dataset}.mat"))
    #graph = scio.loadmat("/datasets/{}.mat".format(dataset))
    #attr = graph['Attributes'] if ('Attributes' in graph) else graph['X']
    if dataset in ['cora', 'ACM','pubmed','citeseer' ,'BlogCatalog','Flickr']:
        attr = graph['Attributes'].A
    else:
        attr = graph['Attributes'] if ('Attributes' in graph) else graph['X']
    #attr = graph['Attributes']
    #adj = graph['Network']
    adj = graph['Network'] if ('Network' in graph) else graph['A']
    #label1 = graph['Label']
    label1 = graph['Label'] if ('Label' in graph) else graph['gnd']
    if dataset in ['cora', 'ACM','pubmed','citeseer' ,'BlogCatalog','Flickr']:
        label = label1  
    else:
        label = label1.T

    # classes = np.squeeze(np.array(graph['Class'],dtype=np.int64) - 1)
    # num_classes = np.max(classes) + 1
    # classes_onehot = dense_to_one_hot(classes, num_classes)

    if 'str_anomaly_label' in graph:
        str_ano_labels = np.squeeze(np.array(graph['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(graph['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_nodes = adj.shape[0]
    num_train = int(num_nodes * train_rate)
    num_val = int(num_nodes * val_rate)
    all_index = list(range(num_nodes))
    random.shuffle(all_index)
    idx_train = all_index[:num_train]
    idx_val = all_index[num_train : num_train+num_val]
    idx_test = all_index[num_train+num_val : ]
    #return attr, adj, label, classes_onehot, str_ano_labels, attr_ano_labels, idx_train, idx_val, idx_test
    return attr, adj, label, str_ano_labels, attr_ano_labels, idx_train, idx_val, idx_test

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    
    return features, sparse_to_tuple(sp.lil_matrix(features))

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def aug_random_edge(input_adj, drop_percent=0.2):
    """
    randomly delect partial edges and
    randomly add the same number of edges in the graph
    """
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()
    num_drop = int(len(row_idx) * percent)

    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))
    drop_idx = random.sample(edge_index, k=num_drop)

    list(map(edges.__delitem__, filter(edges.__contains__, drop_idx)))

    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0]
    new_col_idx = new_edges[1]
    data = np.ones(len(new_row_idx)).tolist()

    new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape=input_adj.shape)

    row_idx, col_idx = (new_adj.todense() - 1).nonzero()
    no_edges_cells = list(zip(row_idx, col_idx))
    add_idx = random.sample(no_edges_cells, num_drop)
    new_row_idx_1, new_col_idx_1 = list(zip(*add_idx))
    row_idx = new_row_idx + new_row_idx_1
    col_idx = new_col_idx + new_col_idx_1
    data = np.ones(len(row_idx)).tolist()

    new_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=input_adj.shape)

    return new_adj


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
def matrix_to_table(martix, dataset, osn_edgelist_weighted_file):

    num_nodes = martix.shape[0]
    num_edges = 0
    with open(osn_edgelist_weighted_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(num_nodes)):
            for j in range(num_nodes):
                if martix[i][j] != 0.0:                               
                    f.write(str(i) + '\t' + str(j) + '\n')
                    num_edges += 1
        print(f'Finished saving {dataset}_edge_list.txt'
              f'\n{num_edges} edges in all')


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:                  
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv


def get_first_adj(dgl_graph, adj, subgraph_size):
    """Generate the first view's subgraph with the first-order neighbor."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj = np.array(adj.todense()).squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])
        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj.append(node_id) 
            subgraphs[node_id].extend(
                list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs


def generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2):
    """Generate subgraph with RWR/first & second -neiborhood algorithm."""
    restart_prob_1 = args.restart_prob_1
    restart_prob_2 = args.restart_prob_2

    if args.subgraph_mode == 'random':
        subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size_1, restart_prob=restart_prob_1)
        subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size_2, restart_prob=restart_prob_2)
    elif args.subgraph_mode == '1+1':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_first_adj(dgl_graph, A, subgraph_size_2)
    elif args.subgraph_mode == '1+2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj(dgl_graph, A, subgraph_size_2)
    else:
        raise NotImplementedError


    return subgraphs_1, subgraphs_2


def get_second_adj(dgl_graph, adj, subgraph_size):
    """Generate the second view's subgraph with the 1/2 first-order and 1/2 second-order neighbor. """
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())
    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)
    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()
    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)
        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()
        if len(first_adj) < subgraph_size // 2:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=True)))
            if len(second_adj) == 0:
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=True)))
            elif len(second_adj) < (subgraph_size - 1) // 2:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))
        else:
            if len(second_adj) == 0:
                first_adj.append(node_id)
                if len(first_adj) < subgraph_size - 1:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=True)))
                else:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=False)))
            elif len(second_adj) < (subgraph_size - 1) // 2 :
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))

        subgraphs[node_id].append(node_id)
    return subgraphs



def center_subgraph_features(attr, subgraphs):
    
    centered_features = []
    for i in range(len(subgraphs)):
        root_feature = attr[i] 
        subgraph_features = attr[subgraphs[i]]  
       
        centered = subgraph_features - root_feature.reshape(1, -1)
        centered_features.append(centered)
    return centered_features





def subgraph_community(adj_dense, args, nb_nodes):

    osn_edgelist_file = './dataset/' + str(args.dataset) + '_edgelist_delete_self_loop.txt'
    if not os.path.exists(osn_edgelist_file):
        
        matrix_to_table(adj_dense, args.dataset, osn_edgelist_file)

    subgraphs = generate_community_subgraph(osn_edgelist_file, nb_nodes, args)

    return subgraphs

def generate_community_subgraph(osn_edgelist_file, nb_nodes, args):
    """Generate subgraph with intro_comunity random walk algorithm."""
    subv = []

    walks_file = './dataset/' + str(args.dataset) + '_walks' + '.npy'
    if not os.path.exists(walks_file):
        walks_model = deep_walk(osn_edgelist_file)
        walks = walks_model.forward()
        walks = np.array(walks)
        np.save(walks_file, walks)
    walks = np.load(walks_file, allow_pickle=True, encoding='latin1')

    subv_index = [i for i in range(20)]
    random.shuffle(subv_index)
    walks_shuffle = walks[nb_nodes*subv_index[0]:nb_nodes*(subv_index[0]+1)]
    for i,walk in enumerate(walks_shuffle):
        walk_unique = np.zeros([len(np.unique(walk))], dtype=int)
        j = 0
        for w in walk:
            if w not in walk_unique:
                walk_unique[j] = w
                j += 1

        subv.append(walk_unique)
        subv[i] = list(subv[i][:args.subgraph_community_size-1])
        subv[i].append(i)                                                  
    return subv