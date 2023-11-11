import argparse
import numpy as np
import random
from model1 import Drugcell
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as auc3
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=36,
                        help='Node dimension')
    parser.add_argument('--lr', type=float, default=0.008)  # 学习率
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='l2 reg')  # 权重衰减参数


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


    rna_seq = np.load('tpm.npy', allow_pickle=True).astype(float)
    rna_seq = sklearn.preprocessing.scale(rna_seq, with_mean=True, with_std=True,axis=0)



    rna_seq = torch.as_tensor(rna_seq, dtype=torch.float32).to(device)



    gene_cnv = np.load('cnv.npy',allow_pickle=True).astype(float)
    gene_cnv = torch.as_tensor(gene_cnv, dtype=torch.float32).to(device)




    pic_dim = np.load('11.npy', allow_pickle=True) #list   259   3x224x224
    A = np.stack(pic_dim, axis=0)
    pic_dim = torch.as_tensor(A, dtype=torch.float32).to(device)


    sensitive = np.load('sensi.npy', allow_pickle=True)
    resistant = np.load('resis.npy', allow_pickle=True)
    sensitive = sensitive[0:13000,:]
    drugcell = np.vstack((resistant, sensitive)).astype(int)

    y = drugcell[:, 2]  # 规定测试标签


    # drug fea,adj
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
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01)).to(device)

    for train1, test1 in fold.split(drugcell, y):

        train_index = drugcell[train1]
        test_index = drugcell[test1]

        train_target = torch.from_numpy(train_index[:,2]).to(device)
        test_target = torch.from_numpy(test_index[:,2]).to(device)

        train_index = torch.from_numpy(train_index).to(device)
        test_index = torch.from_numpy(test_index).to(device)

        model = Drugcell(
                  num_tpm = rna_seq.shape[1],
                  num_genecnv = gene_cnv.shape[1],
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

            loss1,   total_loss2, y, target = model(pic_dim, rna_seq,  gene_cnv,  drug_fea, drug_adj, train_index, train_target, logit_scale)
            loss = 0.6*loss1 + 0.4*total_loss2
            loss.backward()
            optimizer.step()
            acc = (y.argmax(dim=1) == target).sum().type(torch.float) / y.shape[0]
            auc = roc_auc_score(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
            precision, recall, thresholds = precision_recall_curve(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
            aupr = auc3(recall, precision)
            print("Train set results:",
                  "loss1={: .4f}".format(loss1.detach().cpu().numpy()),
                  "loss3={: .4f}".format(total_loss2.detach().cpu().numpy()),
                  "loss_train= {:.4f}".format(loss.detach().cpu().numpy()),
                  "train_auc= {:.4f}".format(auc.item()),
                  "train_aupr= {:.4f}".format(aupr.item()),
                  "train_accuracy= {:.4f}".format(acc.item()))

            model.eval()
            with torch.no_grad():
                '''重写model.forward'''
                loss1,  total_loss2, y, target = model.forward(pic_dim, rna_seq,  gene_cnv, drug_fea, drug_adj, test_index, test_target, logit_scale)
                test_loss =  0.6*loss1 + 0.4*total_loss2
                acc = (y.argmax(dim=1) == target).sum().type(torch.float) / y.shape[0]
                y_pro = y[:, 1]
                auc = roc_auc_score(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
                precision, recall, thresholds = precision_recall_curve(target.detach().cpu().numpy(), y[:, 1].detach().cpu().numpy())
                aupr = auc3(recall, precision)
                print("Test set results:",
                      "loss1={: .4f}".format(loss1.detach().cpu().numpy()),
                      "loss3={: .4f}".format(total_loss2.detach().cpu().numpy()),
                      "loss_test={:.4f}".format(test_loss.detach().cpu().numpy()),
                      "test_auc= {:.4f}".format(auc.item()),
                      "test_aupr= {:.4f}".format(aupr.item()),
                      "test_accuracy= {:.4f}".format(acc.item()))
                if best < (aupr + auc ):
                    best = aupr + auc
                    best_aupr = aupr
                    best_auc = auc
                    best_acc = acc
                    best_pre = y

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

