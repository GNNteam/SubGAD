import numpy as np
import scipy.sparse as sp
from embedding import createWlEmbedding
from tqdm import tqdm


def generate_hnodes(h_adj):
    h_adj = h_adj.tocoo()
    h_index = [[] for i in range(h_adj.shape[0])]
    for i, j in zip(h_adj.row, h_adj.col):
        h_index[i].append(j)
    return h_index


def generate_hadj(adj, h):
    adj_h = sp.eye(adj.shape[0])
    adj_tot = sp.eye(adj.shape[0])
    for i in range(h):
        adj_h = adj_h * adj
        adj_tot = adj_tot + adj_h
    return adj_tot

def generate_scores(scores, M, args):
    # 基于深度的异常评分
    score_weight = [np.math.pow(args.lamda, i) for i in range(6)]
    en_scores = np.zeros_like(scores)
    tot = np.zeros_like(scores)
    for i in range(len(M)):                             # 遍历所有子图
        for key, values in M[i].items():                # 邻居阶数越低， 异常分数的权重越高
            en_scores[key] += score_weight[values] * scores[i]      # 邻居（0阶 + 1阶 + 2阶 + ...）异常分数
            tot[key] += score_weight[values]                        # 源节点总的异常分数
    return np.divide(en_scores, tot)

# Generate h_nodes and their height
def generate_h_nodes_n_dict(adj, h):
    adj_h = sp.eye(adj.shape[0])            #
    M = [{i: 0} for i in range(adj.shape[0])]
    h_index = [[i] for i in range(adj.shape[0])]
    for _ in range(h):
        adj_h = sp.coo_matrix(adj_h * adj)               # 理论：保留原邻接矩阵对角线上有的值?  实际：h=1->保持, h=2, 更新adj_h

        for i, j in zip(adj_h.row, adj_h.col):           # load the index(row and col) of edge
            if j in M[i]:
                continue
            else:
                M[i][j] = _ + 1
                h_index[i].append(j)
    return M, h_index


def generate_subgraph_embeddings(attr, adj, subgraph_index, h):
    embedding = []
    for i in tqdm(range(adj.shape[0])):                                                       # 遍历目标节点， 遍历h阶子图
        root_feature = attr[i, :]
        feature = attr[subgraph_index[i]]                                                     # subgraphv i features
        feature = feature - np.tile(root_feature, (len(subgraph_index[i]), 1))                # subgraphv i features - 目标节点特征的拷贝         "属性特征空间的子图中心化"
        adj_i = adj[subgraph_index[i], :][:, subgraph_index[i]]                               # subgraph adj
        embedding.append(createWlEmbedding(feature, adj_i, h).reshape(1, -1))
    return np.concatenate(embedding, axis=0)                                                  # 公式 (4.2) subgraph embedding 拼接


# def generate_subgraph_embeddings(attr, adj, subgraph_index, h):
#     embedding = []
#     for i in tqdm(range(adj.shape[0])):
#         subgraph_features = attr[subgraph_index[i]]  # 获取子图所有节点特征
#         mean_pooled = np.mean(subgraph_features, axis=0)  # 平均池化
#         centered_features = subgraph_features - mean_pooled  # 用均值中心化替代目标节点中心化
#         adj_i = adj[subgraph_index[i], :][:, subgraph_index[i]]
#         embedding.append(createWlEmbedding(centered_features, adj_i, h).reshape(1, -1))
#     return np.concatenate(embedding, axis=0)

def subgraph_h_embeddings(attr, adj, h):
    M, h_index = generate_h_nodes_n_dict(adj, h)
    embedding = generate_subgraph_embeddings(attr, adj, h_index, h)
    return embedding, M
