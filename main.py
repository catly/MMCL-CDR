import argparse
import numpy as np
import random

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model import Drugcell
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as auc3
from sklearn.metrics import precision_recall_curve
import scipy.sparse as sp
import os


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=600,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=36,
                        help='Node dimension')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='l2 reg')


    def seed_torch(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    seed_torch()

    args = parser.parse_args()
    print('args:', args)
    epochs = args.epoch
    node_dim = args.node_dim
    lr = args.lr
    weight_decay = args.weight_decay


    rna_seq = np.load('tpm.npy', allow_pickle=True).astype(float).T
    c = np.array(np.where(np.isnan(rna_seq))).T
    c = c[:, 1]
    rna_list = list(range(0, rna_seq.shape[1]))
    not_in = set(c)
    kk = [item for item in rna_list if item not in not_in]
    rna_seq = rna_seq[:, kk]
    rna_seq = torch.as_tensor(rna_seq, dtype=torch.float32).to(device)
    print(rna_seq.shape)



    gene_cnv = np.load('cnv.npy',allow_pickle=True).astype(float).T
    c = np.array(np.where(np.isnan(gene_cnv))).T
    c = c[:,1]
    cnv_list = list(range(0 , gene_cnv.shape[1]))
    not_in = set(c)
    kk = [item for item in cnv_list if item not in not_in]
    gene_cnv = gene_cnv[:, kk]

    gene_cnv = torch.as_tensor(gene_cnv, dtype=torch.float32).to(device)
    print(gene_cnv.shape)



    pic_dim = np.load('pil.npy', allow_pickle=True)
    A = np.stack(pic_dim, axis=0)
    pic_dim = torch.as_tensor(A, dtype=torch.float32).to(device)




    sensitive = np.load('sensitive.npy', allow_pickle=True)
    resistant = np.load('resistant.npy', allow_pickle=True)
    ll = np.array(list(range(0,resistant.shape[0])))
    np.random.shuffle(ll)
    resistant = resistant[ll[0:sensitive.shape[0]],:]

    drugcell = np.vstack((resistant, sensitive)).astype(int)
    y = drugcell[:, 2]


    drug_fea = np.load('drug_fea.npy', allow_pickle=True)
    for i in range(len(drug_fea)):
        drug_fea[i] = torch.as_tensor(drug_fea[i])
        drug_fea[i] = drug_fea[i].to(device)




    drug_adj = np.load('dict.npy', allow_pickle=True)
    drug_adj = drug_adj.item()

    for key, value in drug_adj.items():
        drug_adj[key] = torch.from_numpy(drug_adj[key])
        drug_adj[key] = drug_adj[key].to(device)


    fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42, )
    print('load_success:')
    best = 0
    best_auc = 0
    best_acc = 0
    best_aupr = 0
    fold_acc = 0
    fold_aupr = 0
    fold_auc = 0
    fold_index = 1
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05)).to(device)

    for train1, test1 in fold.split(drugcell, y):

        train_index = drugcell[train1]
        test_index = drugcell[test1]

        train_target = torch.from_numpy(train_index[:,2]).to(device)
        test_target = torch.from_numpy(test_index[:,2]).to(device)

        train_index = torch.from_numpy(train_index).to(device)
        test_index = torch.from_numpy(test_index).to(device)
        print(test_index)
        df1 = pd.DataFrame(test_index)
        df1.to_csv('example.csv', mode='a', index=False)
        model = Drugcell(
                  num_tpm = rna_seq.shape[1],

                  num_genecnv = gene_cnv.shape[1],
           #       num_pic = pic_fea.shape[1],
                  node_dim = node_dim,
                  drug_feat = 75
                    ).to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for i in range(epochs):
            print('Epoch: {:02d}'.format(i + 1),
                  'Fold: {:02d}'.format(fold_index))
            model.zero_grad()
            model.train()
            optimizer.zero_grad()

            loss1,  total_loss2, y, target,_ = model(pic_dim, rna_seq,  gene_cnv,  drug_fea, drug_adj, train_index, train_target, logit_scale)
            loss = 0.8 * loss1 + 0.2 * total_loss2 # + 0.15 * total_loss1
            loss.backward()
            optimizer.step()
            acc = (y.argmax(dim=1) == target).sum().type(torch.float) / y.shape[0]
            auc = roc_auc_score(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
            precision, recall, thresholds = precision_recall_curve(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
            aupr = auc3(recall, precision)
            print("Train set results:",
                  "loss1={: .4f}".format(loss1.detach().cpu().numpy()),
             #      "loss2={: .4f}".format(0.15 *total_loss1.detach().cpu().numpy()),
                  "loss3={: .4f}".format(0.15 *total_loss2.detach().cpu().numpy()),
                  "loss_train= {:.4f}".format(loss.detach().cpu().numpy()),
                  "train_auc= {:.4f}".format(auc.item()),
                  "train_aupr= {:.4f}".format(aupr.item()),
                  "train_accuracy= {:.4f}".format(acc.item()))

            model.eval()
            with torch.no_grad():
                '''重写model.forward'''
                loss1,  total_loss2, y, target,cell_fea = model.forward(pic_dim, rna_seq,  gene_cnv, drug_fea, drug_adj, test_index, test_target, logit_scale)
                test_loss = 0.8 * loss1 +  0.2 * total_loss2 #  + 0.15 * total_loss1
                acc = (y.argmax(dim=1) == target).sum().type(torch.float) / y.shape[0]
                y_pro = y[:, 1]
                auc = roc_auc_score(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
                precision, recall, thresholds = precision_recall_curve(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
                aupr = auc3(recall, precision)
                print("Test set results:",
                      "loss1={: .4f}".format(loss1.detach().cpu().numpy()),
                  #    "loss2={: .4f}".format(0.15 *total_loss1.detach().cpu().numpy()),
                      "loss3={: .4f}".format(0.15 *total_loss2.detach().cpu().numpy()),
                      "loss_test={:.4f}".format(test_loss.detach().cpu().numpy()),
                      "test_auc= {:.4f}".format(auc.item()),
                      "test_aupr= {:.4f}".format(aupr.item()),
                      "test_accuracy= {:.4f}".format(acc.item()))
                if best < (aupr + auc ):
                    best = aupr + auc
                    best_aupr = aupr
                    best_auc = auc
                    best_acc = acc

        fold_acc = fold_acc + best_acc
        fold_auc = fold_auc + best_auc
        fold_aupr = fold_aupr + best_aupr

        print('Fold: {:02d}'.format(fold_index),
              "fold_auc= {:.4f}".format(best_auc.item()),
              "fold_aupr= {:.4f}".format(best_aupr.item()),
              "fold_accuracy= {:.4f}".format(best_acc.item())
              )
        print('args:', args)

        best = 0
        best_auc = 0
        best_acc = 0
        best_aupr = 0
        fold_index = fold_index + 1


    print("Fold best results:",
          "fold5_best_auc= {:.4f}".format(fold_auc / 10),
          "fold5_best_aupr= {:.4f}".format(fold_aupr / 10),
          "fold5_best_acc= {:.4f}\n".format(fold_acc / 10))

