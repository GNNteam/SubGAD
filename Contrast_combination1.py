import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from topology_dist import *
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from utility import load_graph, subgraph_community, adj_to_dgl_graph, generate_rwr_subgraph, parser, set_seed, \
    preprocess_features, normalize_adj, aug_random_edge,generate_subgraph,center_subgraph_features
import argparse
from subgraph_centralization import subgraph_h_embeddings, generate_scores
from sklearn.metrics import roc_auc_score
from iNN_IK import iNN_IK

from model1 import Model

import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler



print("Starting training...")
start_train_time = time.time()
def train_SUBGAD1(args):

    #s1=time.time()
    #device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    attr, adj0, label, \
     str_ano_labels, attr_ano_labels, idx_train, idx_val, idx_test = load_graph(args.dataset)

    # data process
    features, _ = preprocess_features(attr)
    adj0 = normalize_adj(adj0)
    adj0 = (adj0 + sp.eye(adj0.shape[0]))
    A = adj0
    full_topology_dist = register_topology(args.dataset, adj0)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    raw_feature = torch.FloatTensor(attr[np.newaxis])
    
    adj_ = adj0

    adj = sp.csr_matrix(adj0)
    dgl_graph = adj_to_dgl_graph(adj)
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = torch.FloatTensor(adj[np.newaxis])


    # adj0 = normalize_adj(adj0).toarray()
    # adj0 = torch.FloatTensor(adj0[np.newaxis]).to(device)
    features = torch.FloatTensor(features[np.newaxis]).to(device)



    # (1) subgraph VS subgraph
    embedding, M = subgraph_h_embeddings(attr, adj_, args.h)
    kmembeddings = iNN_IK(args.psi, 100).fit_transform(embedding)
    mean_embedding = np.mean(kmembeddings, axis=0)

    # point detector
    scores = kmembeddings.dot(mean_embedding.transpose())
    final_scores = generate_scores(scores, M, args)
    auc1 = roc_auc_score(label, -np.asarray(final_scores))



    score_sub_sub = -np.asarray(final_scores)
    #np.save(final_scores, score_sub_sub)

    print(
        f'dataset : {args.dataset}, psi={args.psi}, h={args.h}, lamda={args.lamda}, auc = {auc1}\n')
    #score_sub_sub = MinMaxScaler.fit_transform(score_sub_sub)
    scaler1 = MinMaxScaler()
    ano_score_ss =scaler1.fit_transform(score_sub_sub.reshape(-1, 1)).reshape(-1)
    


    print(f'"noed VS subgraph" and "node VS node"...')
    batch_size = args.batch_size
    subgraph_size_1 = args.K_1
    subgraph_size_2 = args.K_2

  
    alpha_inter = args.alpha
    beta = args.beta
    print(f"Using alpha={alpha_inter}, beta={beta}")

    # Initialize model and optimiser
    model = Model(n_in=ft_size, n_h=args.embedding_dim, activation='prelu', negsamp_round=args.negsamp_ratio,
                readout=args.readout, hidden_size=args.hidden_size, temperature=args.temperature,
                Sinkhorn_iter_times = args.Sinkhorn_iter_times, lamb=args.Sinkhorn_lamb, is_rectified=args.rectified,
                topo_t = args.topo_t, have_neg = args.have_neg, neg_top_k = args.neg_top_k)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        full_topology_dist = full_topology_dist.to(device)
        model.to(device)
        features = features.to(device)
        raw_feature = raw_feature.to(device)
        adj = adj.to(device)
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    mse_loss = nn.MSELoss(reduction='mean')


    if nb_nodes % batch_size == 0:
        batch_num = nb_nodes // batch_size
    else:
        batch_num = nb_nodes // batch_size + 1



    # # Train model
    with tqdm(total=args.epochs) as pbar:
        pbar.set_description('Training')

        for epoch in range(args.epochs):

            model.train()

            all_idx = list(range(nb_nodes))

            random.shuffle(all_idx)


            loss_1 = 0.
            loss_2 = 0.
            loss_3 = 0.
            loss_record = 0.

            total_loss = 0.
            

            subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2)


            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

                ba = []
                ba_2 = []
                bf = []
                bf_2 = []
                raw = []
                raw_2 = []
                subgraph_idx = []
                subgraph_idx_2 = []

                Z_l = torch.full((cur_batch_size,), 1.)
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
                added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
                added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_adj_zero_col_2[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    Z_l = Z_l.to(device)
                    lbl = lbl.to(device)
                    added_adj_zero_row = added_adj_zero_row.to(device)
                    added_adj_zero_col = added_adj_zero_col.to(device)
                    added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                    added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                    added_feat_zero_row = added_feat_zero_row.to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    cur_feat = features[:, subgraphs_1[i], :]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                    raw_f = raw_feature[:, subgraphs_1[i], :]
                    raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                    ba.append(cur_adj)
                    ba_2.append(cur_adj_2)
                    bf.append(cur_feat)
                    bf_2.append(cur_feat_2)
                    raw.append(raw_f)
                    raw_2.append(raw_f_2)
                    subgraph_idx.append(subgraphs_1[i])
                    subgraph_idx_2.append(subgraphs_2[i])

                ba = torch.cat(ba)
                ba_2 = torch.cat(ba_2)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
                ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)


                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                bf_2 = torch.cat(bf_2)
                bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)

                raw = torch.cat(raw)
                raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
                raw_2 = torch.cat(raw_2)
                raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)

                subgraph_idx = torch.Tensor(subgraph_idx)
                subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
                subgraph_idx = subgraph_idx.int()
                subgraph_idx_2 = subgraph_idx_2.int()
                if torch.cuda.is_available():
                    subgraph_idx = subgraph_idx.to(device)
                    subgraph_idx_2 = subgraph_idx_2.to(device)

                #/---------------------MODEL-----------------------/#
                
                disc_1, disc_2, inter_loss_1, inter_loss_2, _, _, _, _ = \
                    model(bf, ba, raw, subgraph_size_1 - 1, bf_2, ba_2, raw_2, subgraph_size_2 - 1,
                            full_topology_dist, subgraph_idx, subgraph_idx_2)
                

                intra_loss_1 = b_xent(disc_1, lbl)
                intra_loss_2 = b_xent(disc_2, lbl)

                #loss_recon = 0.5 * (mse_loss(node_recons_1, raw[:, -1, :]) + mse_loss(node_recons_2, raw_2[:, -1, :]))
                loss_intra = torch.mean((intra_loss_1 + intra_loss_2) / 2)

                loss_inter = torch.mean((inter_loss_1 + inter_loss_2) / 2)

                loss = (1-alpha_inter) * loss_intra + alpha_inter * loss_inter 

                loss.backward()
                optimiser.step()



                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss


            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes


            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'pkl1/best_' + args.dataset + '.pkl')
            else:
                cnt_wait += 1


            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)

    # # Inference phase test
    print('Loading {}th epoch'.format(best_t))
    path = 'pkl1/best_' + args.dataset + '.pkl'
    model.load_state_dict(torch.load(path))
    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    print(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")

    start_infer_time = time.time() 
    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(args.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, A,subgraph_size_1, subgraph_size_2)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba = []
                bf = []
                bf_2 = []
                ba_2 = []
                raw = []
                raw_2 = []
                subgraph_idx = []
                subgraph_idx_2 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size_1))
                added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, subgraph_size_2))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size_1 + 1, 1))
                added_adj_zero_col_2 = torch.zeros((cur_batch_size, subgraph_size_2 + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_adj_zero_col_2[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    added_adj_zero_row = added_adj_zero_row.to(device)
                    added_adj_zero_row_2 = added_adj_zero_row_2.to(device)
                    added_adj_zero_col = added_adj_zero_col.to(device)
                    added_adj_zero_col_2 = added_adj_zero_col_2.to(device)
                    added_feat_zero_row = added_feat_zero_row.to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_adj2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    cur_feat = features[:, subgraphs_1[i], :]
                    raw_f = raw_feature[:, subgraphs_1[i], :]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                    raw_f_2 = raw_feature[:, subgraphs_2[i], :]
                    ba.append(cur_adj)
                    ba_2.append(cur_adj2)
                    bf.append(cur_feat)
                    bf_2.append(cur_feat_2)
                    raw.append(raw_f)
                    raw_2.append(raw_f_2)
                    subgraph_idx.append(subgraphs_1[i])
                    subgraph_idx_2.append(subgraphs_2[i])

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_2 = torch.cat(ba_2)
                ba_2 = torch.cat((ba_2, added_adj_zero_row_2), dim=1)
                ba_2 = torch.cat((ba_2, added_adj_zero_col_2), dim=2)

                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                bf_2 = torch.cat(bf_2)
                bf_2 = torch.cat((bf_2[:, :-1, :], added_feat_zero_row, bf_2[:, -1:, :]), dim=1)
                raw = torch.cat(raw)
                raw = torch.cat((raw[:, :-1, :], added_feat_zero_row, raw[:, -1:, :]), dim=1)
                raw_2 = torch.cat(raw_2)
                raw_2 = torch.cat((raw_2[:, :-1, :], added_feat_zero_row, raw_2[:, -1:, :]), dim=1)

                subgraph_idx = torch.Tensor(subgraph_idx)
                subgraph_idx_2 = torch.Tensor(subgraph_idx_2)
                subgraph_idx = subgraph_idx.int()
                subgraph_idx_2 = subgraph_idx_2.int()
                if torch.cuda.is_available():
                    subgraph_idx = subgraph_idx.to(device)
                    subgraph_idx_2 = subgraph_idx_2.to(device)

                # /---------------------MODEL-----------------------/#

                with torch.no_grad():
                    logits_1, logits_2, inter_loss_1, inter_loss_2, sim_all_1, sim_all_2, \
                    sim_pos_1, sim_pos_2 = model(bf, ba, raw, subgraph_size_1 - 1, bf_2, ba_2, raw_2, subgraph_size_2 - 1,
                                                    full_topology_dist, subgraph_idx, subgraph_idx_2)


                    logits_1 = torch.squeeze(logits_1)
                    logits_1 = torch.sigmoid(logits_1)

                    logits_2 = torch.squeeze(logits_2)
                    logits_2 = torch.sigmoid(logits_2)

                pdist = nn.PairwiseDistance(p=2)
                scaler1 = MinMaxScaler()
                scaler3 = MinMaxScaler()


                score_co1 = - (logits_1[:cur_batch_size] - logits_1[cur_batch_size:]).cpu().numpy()
                score_co2 = - (logits_2[:cur_batch_size] - logits_2[cur_batch_size:]).cpu().numpy()
                score_co = (score_co1 + score_co2) / 2

                score_ot = - (sim_pos_1 + sim_pos_2) / 2
                score_ot = score_ot.cpu().numpy()

                #nomalize
                ano_score_co = scaler1.fit_transform(score_co.reshape(-1, 1)).reshape(-1)
    
                ano_score_ot = scaler3.fit_transform(score_ot.reshape(-1, 1)).reshape(-1)
                ano_score = (1-alpha_inter) * ano_score_co + alpha_inter * ano_score_ot
                multi_round_ano_score[round, idx] = ano_score
            pbar_test.update(1)

    ano_score_final = np.mean(multi_round_ano_score, axis=0)
    auc2 = roc_auc_score(label, ano_score_final)
    
    end_infer_time = time.time()
    infer_time = end_infer_time - start_infer_time
    print(f"Inference completed in {infer_time:.4f} seconds.")


    ano_scores = ano_score_ss * beta + (1 - beta) * ano_score_final
    AUC = roc_auc_score(label, ano_scores)
    AUC_value = float(f"{AUC:.4f}")
    
    print(f"alpha={alpha_inter}, beta={beta}, AUC: {AUC:.4f}")
    
    return AUC_value
            
           




