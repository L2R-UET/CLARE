import torch
import torch.nn as nn, torch.nn.functional as F
import math
from tqdm import tqdm
from utils.dataset import Dataset
from torch_scatter import scatter
from torch_geometric.utils import softmax

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class Argument:
    def __init__(self, args_dict):
        self.edge_num = args_dict['edge_num']
        self.out_nhead = args_dict['out_nhead']


class hhgnnConv(nn.Module):
    def __init__(self, args, edge_num, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=False, device='cuda'):
        super().__init__()
        self.device = device

        self.W = nn.Linear(in_channels, heads * out_channels, bias=True).to(self.device)
        
        self.att_v_user=nn.Parameter(torch.Tensor(1, heads, out_channels)).to(self.device)
        self.att_v_item = nn.Parameter(torch.Tensor(1, heads, out_channels)).to(self.device)
        self.att_v_aspect = nn.Parameter(torch.Tensor(1, heads, out_channels)).to(self.device)

        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels)).to(self.device)

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout).to(self.device)
        self.leaky_relu = nn.LeakyReLU(negative_slope).to(self.device)
        self.skip_sum = skip_sum
        self.edge_num=edge_num
        self.reset_parameters()

        self.relu = nn.ReLU()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels, self.out_channels, self.heads)
    def reset_parameters(self):

        glorot(self.att_v_user)
        glorot(self.att_v_item)
        glorot(self.att_v_aspect)
        glorot(self.att_e)
        
    def forward(self, X, vertex, edges, V_class_index, V_class_index_aspect, V_class_index_user, V_class_index_item):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        X0 = self.W(X)

        X = X0.view(N, H, C)
        Xve = X[vertex].to(self.device)
        X = Xve
        
        X_e = (X * self.att_e).sum(-1)
        beta = self.leaky_relu(X_e).to(self.device)
        beta = softmax(beta, edges, num_nodes=self.edge_num)
        beta = beta.unsqueeze(-1)
        Xe = Xve * beta
        Xe = (scatter(Xe, edges, dim=0, reduce='sum', dim_size=self.edge_num))

        Xe = Xe[edges]
        Xe_aspect = (torch.index_select(Xe, 0, V_class_index_aspect) * self.att_v_aspect).sum(-1)
        Xe_user = (torch.index_select(Xe, 0, V_class_index_user) * self.att_v_user).sum(-1)
        Xe_item = (torch.index_select(Xe, 0, V_class_index_item) * self.att_v_item).sum(-1)
        Xe_all = torch.cat((Xe_aspect, Xe_user, Xe_item), 0)
        alpha_e = torch.gather(Xe_all, 0, V_class_index)
        alpha = self.leaky_relu(alpha_e)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = alpha.unsqueeze(-1)
        Xev = Xe * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)  # [N, H, C]
        Xv = Xv.view(N, H * C)
        Xv = self.relu(Xv)

        return Xv

class HHGNN(nn.Module):
    def __init__(self, args, dataset: Dataset, nfeat, nhid, out_dim, out_nhead, nhead, node_input_dim):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.conv_out = hhgnnConv(args, len(dataset.trn_csv), nhid * nhead, nhid, heads=out_nhead, device=self.device)
        self.conv_in = hhgnnConv(args, len(dataset.trn_csv), nfeat, nhid, heads=nhead, device=self.device)

        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act['relu']

        self.dataset = dataset
        self.node_input_dim=node_input_dim

        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear(nhid * out_nhead, out_dim, bias=True).to(self.device)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True) for feats_dim in node_input_dim])

        self.user_embeddings = nn.Embedding(self.dataset.n_user, out_dim).to(self.device)
        self.item_embeddings = nn.Embedding(self.dataset.n_item, out_dim).to(self.device)
        self.aspect_embeddings = nn.Embedding(self.dataset.n_aspect, out_dim).to(self.device)

        self.init_weight()
        self.hypergraph_construction()

    def init_weight(self):
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.aspect_embeddings.weight)

    def hypergraph_construction(self):
        self.train_tuples, self.test_tuples = [], []
        
        for _, row in tqdm(self.dataset.trn_csv.iterrows()):
            if len(row['aspect_id']) > 0:
                hyperedge = []
                aspect_hyperedge = row['aspect_id']
            
                hyperedge.extend(aspect_hyperedge)
                hyperedge.append(self.dataset.user_to_index[row['uid']] + self.dataset.user_offset)
                hyperedge.append(self.dataset.item_to_index[row['iid']] + self.dataset.item_offset)
                self.train_tuples.append(tuple(hyperedge))
        
        self.V = []
        self.E = []
        V_class = []
        for e_id, t in enumerate(self.train_tuples):
            for node in t:
                self.V.append(node)
                if node < self.dataset.user_offset:
                    V_class.append(0) # 0 represents aspect
                elif node < self.dataset.item_offset:
                    V_class.append(1) # 1 represents user
                else:
                    V_class.append(2) # 2 represents item
                self.E.append(e_id)
        
        self.V_class_index = torch.tensor([list(range(len(V_class))), 
                                           list(range(len(V_class))), 
                                           list(range(len(V_class)))]).T.to(self.device)
        self.V_class_index_aspect = torch.tensor([i for i, x in enumerate(V_class) if x == 0]).to(self.device)
        self.V_class_index_user = torch.tensor([i for i, x in enumerate(V_class) if x == 1]).to(self.device)
        self.V_class_index_item = torch.tensor([i for i, x in enumerate(V_class) if x == 2]).to(self.device)
        self.hyperedge_index = torch.tensor([self.V, self.E], dtype=torch.long)

        self.V = torch.tensor(self.V).to(self.device)
        self.E = torch.tensor(self.E).to(self.device)

        self.max_len = max(len(t) - 2 for t in self.train_tuples)
    
        # Prepare padded aspect tensors and masks
        u_list, i_list, a_lists, masks = [], [], [], []
        
        u_list_test, i_list_test = [], []
        
        for t in self.train_tuples:
            *a_list, u, i = t
            pad_len = self.max_len - len(a_list)
            a_lists.append(a_list + [-1] * pad_len)  # Use -1 as a padding value
            masks.append([1] * len(a_list) + [0] * pad_len)
            u_list.append(u)
            i_list.append(i)
        
        for t in self.test_tuples:
            *a_list, u, i = t
            u_list_test.append(u)
            i_list_test.append(i)
        
        # Convert to tensors
        self.u_tensor = torch.tensor(u_list, device=self.device)
        self.i_tensor = torch.tensor(i_list, device=self.device)
        self.a_tensor = torch.tensor(a_lists, device=self.device)
        self.mask_tensor = torch.tensor(masks, device=self.device)

    def forward(self):
        user_emb = self.user_embeddings.weight
        item_emb = self.item_embeddings.weight
        aspect_emb = self.aspect_embeddings.weight
        
        X = torch.cat([aspect_emb, user_emb, item_emb], dim=0).to(self.device)
        X = self.conv_in(X, self.V, self.E, 
                         self.V_class_index, 
                         self.V_class_index_aspect, 
                         self.V_class_index_user, 
                         self.V_class_index_item)
        X = self.conv_out(X, self.V, self.E, 
                          self.V_class_index, 
                          self.V_class_index_aspect, 
                          self.V_class_index_user, 
                          self.V_class_index_item)
        X = self.lin_out1(X)
    
        # Flatten aspects and masks
        a_pos = self.a_tensor.view(-1)
        valid_mask = (a_pos != -1)
    
        # Apply the mask
        a_pos = a_pos[valid_mask]
        u_rep = self.u_tensor.repeat_interleave(self.max_len)[valid_mask]
        i_rep = self.i_tensor.repeat_interleave(self.max_len)[valid_mask]
    
        # Sample negative aspects
        a_neg = torch.randint(0, self.dataset.n_aspect, (len(a_pos),), device=self.device)
        mask_neg = (a_neg == a_pos)
        while mask_neg.any():
            a_neg[mask_neg] = torch.randint(0, self.dataset.n_aspect, (mask_neg.sum().item(),), device=self.device)
            mask_neg = (a_neg == a_pos)
    
        # Calculate loss
        a_pos_emb = X[a_pos]
        a_neg_emb = X[a_neg]
        u_emb_rep = X[u_rep]
        i_emb_rep = X[i_rep]
        loss = self.bpr_loss(u_emb_rep, i_emb_rep, a_pos_emb, a_neg_emb).mean()
        
        return loss

    def bpr_loss(self, u_emb, i_emb, a_pos_emb, a_neg_emb):
        ui_emb = u_emb + i_emb
        pos_score = F.cosine_similarity(ui_emb, a_pos_emb, dim=1)
        neg_score = F.cosine_similarity(ui_emb, a_neg_emb, dim=1)
        diff = pos_score - neg_score
        return -F.logsigmoid(diff)

    def evaluate(self):
        user_emb = self.user_embeddings.weight
        item_emb = self.item_embeddings.weight
        aspect_emb = self.aspect_embeddings.weight
        
        X = torch.cat([aspect_emb, user_emb, item_emb], dim=0).to(self.device)
        X = self.conv_in(X, self.V, self.E, 
                         self.V_class_index, 
                         self.V_class_index_aspect, 
                         self.V_class_index_user, 
                         self.V_class_index_item)
        X = self.conv_out(X, self.V, self.E, 
                          self.V_class_index, 
                          self.V_class_index_aspect, 
                          self.V_class_index_user, 
                          self.V_class_index_item)
        X = self.lin_out1(X)
        return X