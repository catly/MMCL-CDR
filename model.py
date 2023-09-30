import torch
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
import torch.nn as nn
from torchvision.models import resnet50




device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
class Drugcell(nn.Module):
    def __init__(self,
                 num_tpm,
      #           num_dname,
                 num_genecnv,

                 node_dim,
                 drug_feat
                 ):
        super(Drugcell, self).__init__()

        # u相关参数
        self. num_tpm =  num_tpm

        self.num_genecnv = num_genecnv
    #   self.num_pic = num_pic
        self.node_dim = node_dim
        self.drug_feat = drug_feat


        self.mlp_1 = MLP_1(num_tpm, node_dim)
        self.mlp_2 = MLP_2(num_genecnv, node_dim)
        self.image_model =  ImageModel(node_dim)
    #   self.drug_gcn = GCN_NET(drug_feat, 45 , node_dim*4)

        self.mlp_end = MLP_end(node_dim * 4, 2)
        self.weight = nn.Parameter(torch.Tensor(drug_feat, node_dim*4)).to(device)  # GCN
        self.weight2 = nn.Parameter(torch.Tensor(node_dim*4, node_dim*2)).to(device)  # GCN
        self.reset_parameters()
        self.att = SemanticAttention(36)
        self.loss = nn.CrossEntropyLoss()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight2)



    def gcn_conv(self,X,H):  # 自己写了一个GCN
        X = torch.mm(X, self.weight)  # X-features; self.weight-weight
        H = self.norm(H, add=True)  # H-第i个channel下邻接矩阵;
        X_1 = torch.mm(H.t(), X)
        X_2 = torch.mm(X_1, self.weight2)
        return torch.mm(H.t(), X_2)


    def norm(self, H, add=False):
        H = H.t()   # t
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))  # 建立一个对角阵; 除了自身节点，对应位置相乘。Degree(排除本身)
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)).to(device) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(device)
        deg = torch.sum(H, dim=1)  # 按行求和, 即每个节点的dgree的和
        deg_inv = deg.pow(-1).to(device)   # deg-1 归一化操作
        deg_inv[deg_inv == float('inf')] = 00
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor).to(device)  # 转换成n*n的矩阵
        H = torch.mm(deg_inv,H)  # 矩阵内积
        H = H.t()
        return H



    def forward(self, pic_dim, rna_seq,  gene_cnv,  drug_fea, drug_adj,  index_list, target, logit_scale):  # model(A_u, node_features_u,  A_v,node_features_v, train_index)

        rna_seq = torch.softmax(rna_seq,dim=0)
        rna_seq1 = self.mlp_1(rna_seq)


        gene_cnv = torch.softmax(gene_cnv, dim=0)
        gene_cnv1 = self.mlp_2(gene_cnv)

        rna_seq1 = rna_seq1 / rna_seq1.norm(dim=1, keepdim=True)
        gene_cnv1 = gene_cnv1 / gene_cnv1.norm(dim=1, keepdim=True)
        rna_fea = logit_scale * rna_seq1  @ gene_cnv1.t()
        cnv_fea = rna_fea.t()
        labels = torch.arange(rna_seq1.shape[0]).long().to(device)
        total_loss1 = (
                              F.cross_entropy(rna_fea, labels) +
                              F.cross_entropy(cnv_fea, labels)
                      ) / 2




        pic_fea1  = self.image_model(pic_dim)
        ccc = torch.stack([rna_seq1,  gene_cnv1],dim=1)
        cell_fea = self.att(ccc)



        pic_fea1 = pic_fea1 / pic_fea1.norm(dim=1, keepdim=True)
        cell_fea = cell_fea / cell_fea.norm(dim=1, keepdim=True)
        image_fea = logit_scale * pic_fea1 @ cell_fea.t()
        text_fea = image_fea.t()

        labels = torch.arange(image_fea.shape[0]).long().to(device)
        total_loss2 = (
                              F.cross_entropy(image_fea, labels) +
                              F.cross_entropy(text_fea, labels)
                      ) / 2



        cell_fea = torch.cat((cell_fea, pic_fea1), dim=1)        #直接拼接
     #  cell_fea = (cell_fea + pic_fea1) / 2                      #相加除以2
     #   ccc = torch.stack([cell_fea, pic_fea1], dim=1)
     #   cell_fea = self.att(ccc)

        temp = 0


        for i in range(len(drug_fea)):

            gcn_fea = F.relu(self.gcn_conv(drug_fea[i].to(torch.float32), drug_adj[i].to(torch.float32)))

            gcn_fea = torch.unsqueeze(torch.mean(gcn_fea,dim=0), dim=-1)

            if temp == 0:
                drugfea = gcn_fea
            else:
                drugfea = torch.cat((drugfea, gcn_fea), dim=1)
            temp = 1

        drug_fea = drugfea.T
        B = torch.cat((cell_fea[index_list[:,0],:], drug_fea[index_list[:,1],:].float()), 1)


        B = self.mlp_end(B)
        B = torch.squeeze(B)

        loss1 = self.loss(B, target)

        return  loss1,  total_loss2, B, target,cell_fea




class MLP_1(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_1, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 4 , input_size // 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 32, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_2, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 4, input_size//32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 32, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out



class ImageModel(nn.Module):

    def __init__(self, output_size):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            nn.Flatten()
        )
        self.trans = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.full_resnet.fc.in_features, output_size),
            nn.Softmax(dim=-1)
        )


    def forward(self, imgs):
        feature = self.resnet(imgs)

        return self.trans(feature)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)




class MLP_end(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_end, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 4, output_size),

            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
