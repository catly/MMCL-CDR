import torch
import torch.nn.functional as F
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Drugcell(nn.Module):
    def __init__(self,
                 num_tpm,
                 num_genecnv,
                 node_dim,
                 drug_feat
                 ):
        super(Drugcell, self).__init__()


        self. num_tpm =  num_tpm
        self.num_genecnv = num_genecnv
        self.node_dim = node_dim
        self.drug_feat = drug_feat

        self.mlp_1 = MLP_1(num_tpm, node_dim)
        self.mlp_2 = MLP_2(num_genecnv, node_dim)

        self.image_model =  ImageModel()
        self.att = SemanticAttention(36)
        self.mlp_end = MLP_end(node_dim * 4, 2)
        self.weight = nn.Parameter(torch.Tensor(drug_feat, node_dim*8)).to(device)  # GCN
        self.weight2 = nn.Parameter(torch.Tensor(node_dim*8, node_dim*2)).to(device)  # GCN
        self.reset_parameters()

        self.loss = nn.CrossEntropyLoss()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight2)



    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        X_1 = torch.mm(H.t(), X)
        X_2 = torch.mm(X_1, self.weight2)
        return torch.mm(H.t(), X_2)


    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)).to(device) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(device)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1).to(device)
        deg_inv[deg_inv == float('inf')] = 00
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor).to(device)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H



    def forward(self, pic_dim, rna_seq,  gene_cnv,  drug_fea, drug_adj,  index_list, target, logit_scale):


        rna_seq1 = self.mlp_1(rna_seq)
        gene_cnv1 = self.mlp_2(gene_cnv)

        pic_fea1  = self.image_model(pic_dim)

        cell_fea = torch.stack((rna_seq1,  gene_cnv1),dim= 1)
        cell_fea = self.att(cell_fea)


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

        temp = 0


        for i in range(len(drug_fea)):

            gcn_fea = F.relu(self.gcn_conv(drug_fea[i].to(torch.float32), drug_adj[i].to(torch.float32)))
            gcn_fea = torch.unsqueeze(torch.max(gcn_fea, dim=0)[0], dim=-1)   # max pooling

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

        return  loss1,  total_loss2, B, target




class MLP_1(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_1, self).__init__()
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


class MLP_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_2, self).__init__()
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

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 128 * 128, 128 * 8),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128 * 8, 36)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



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
