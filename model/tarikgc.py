import torch
from torch import nn
import numpy as np
from model.attention import get_attn_pad_mask,Encoder,last_layer
from model.graph_aggregation import  NodeLayer, EdgeLayer

class Residual(nn.Module):
    """
    A residual layer that adds the output of a function to its input.

    Args:
        fn (nn.Module): The function to be applied to the input.

    """

    def __init__(self, fn):
        """
        Initialize the Residual layer with a given function.

        Args:
            fn (nn.Module): The function to be applied to the input.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """
        Forward pass of the Residual layer.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: The input tensor added to the result of applying the function `fn` to it.
        """
        return x + self.fn(x)


class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block with one fully connected layer.

    Args:
        in_features (int): The number of input features.
        output_size (int): The number of output features.
        bias (boolean): Add bias to the linear layer
        layer_norm (boolean): Apply layer normalization
        dropout (float): The dropout value
        activation (nn.Module): The activation function to be applied after each fully connected layer.

    Example:
    ```python
    # Create an MLP block with 2 hidden layers and ReLU activation
    mlp_block = MLPBlock(input_size=64, output_size=10, activation=nn.ReLU())

    # Apply the MLP block to an input tensor
    input_tensor = torch.randn(32, 64)
    output = mlp_block(input_tensor)
    ```
    """
    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class TarIKGC(nn.Module):
    def __init__(self, num_ent, num_rel, num_base, init_dim, embed_dim,
                 smi_dict, params,mol_feature,protein_go, features_length, disease_feature):

        super(TarIKGC, self).__init__()
        self.p = params
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim,  self.embed_dim = init_dim,  embed_dim
        self.features_length = features_length
        self.init_embed = get_param([self.num_ent, self.init_dim])
        self.init_rel = get_param([self.num_rel * 2, self.init_dim])

        self.smi_dict = smi_dict
        self.bn = torch.nn.BatchNorm1d(self.p.init_dim)
        self.bias = nn.Parameter(torch.zeros(self.p.num_targets+self.p.num_diseases))

        # Transformer Block
        self.mol_feature = mol_feature
        self.enc = Encoder(cpd_atom=78, d_model=self.p.init_dim, n_heads=8, dropout=0.1, distance_matrix_kernel='softmax',
                           d_ffn=128, activation_fun='softmax')
        self.last_layer = last_layer(d_model=self.p.init_dim, dropout=0.1, n_output=1)

        ## GO_embedding
        self.protein_go = protein_go.to("cuda")
        net = []
        net.append(MLPBlock(self.features_length , self.embed_dim ))
        net.append(Residual(MLPBlock(self.embed_dim , self.embed_dim))) # From deepgo2
        self.net = nn.Sequential(*net)

        disease_net = []
        disease_net.append(MLPBlock(1024, self.embed_dim))
        disease_net.append(Residual(MLPBlock(self.embed_dim, self.embed_dim)))
        self.disease_net = nn.Sequential(*disease_net)
        self.disease_feature = disease_feature.to("cuda")
        self.disease_init_emb = get_param([self.p.num_diseases, self.init_dim])

        self.w1 = nn.Parameter(torch.Tensor([0.5]))
        self.w2 = nn.Parameter(torch.Tensor([0.5]))
        self.w3 = nn.Parameter(torch.Tensor([0.5]))

        self.kg_n_layer = 1
        self.edge_layers = nn.ModuleList([EdgeLayer(self.init_dim,params,num_ent, num_rel) for _ in range(self.kg_n_layer)])
        self.node_layers = nn.ModuleList([NodeLayer(self.init_dim,params,num_ent, num_rel) for _ in range(self.kg_n_layer)])
        self.ent_drop = nn.Dropout(self.p.ent_drop)
        self.rel_drop = nn.Dropout(self.p.rel_drop)

        self.loss1 = nn.BCELoss()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.p.neg_ratio)) #


    def calc_loss(self, pred, label):
        return self.loss(pred, label)


    def forward(self, g, subj, rel):
        all_ent, rel_emb = self.aggragate_emb(g)
        rel_emb = torch.index_select(rel_emb, 0, rel)
        self.target_emb= self.target_base()
        self.mol_emb_sub = self.mol_base(subj)
        self.disease_emb = self.disease_base()
        sub_emb, target_emb, disease_emb = self.emb_fusion(self.mol_emb_sub, self.target_emb, self.disease_emb, all_ent, subj)
        obj_emb = sub_emb * rel_emb
        x = torch.mm(obj_emb, torch.cat([target_emb, disease_emb]).transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return x, score

    def mol_base(self, subj):
        drugid = subj[subj < self.p.num_drugs].tolist()
        mol_feature = [self.mol_feature[entity_id] for entity_id in drugid]
        cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix = zip(*mol_feature)
        cpd_atom_features = torch.tensor(np.array(cpd_atom_features), dtype=torch.float32).to("cuda")
        cpd_adj_matrix = torch.tensor(np.array(cpd_adj_matrix), dtype=torch.float32).to("cuda")
        cpd_dist_matrix = torch.tensor(np.array(cpd_dist_matrix), dtype=torch.float32).to("cuda")
        cpd_self_attn_mask = get_attn_pad_mask(cpd_atom_features, cpd_atom_features)
        cpd_enc_output,  cpd_enc_attn_list = self.enc(
            cpd_atom_features=cpd_atom_features, cpd_adj_matrix=cpd_adj_matrix, cpd_dist_matrix=cpd_dist_matrix,
            cpd_self_attn_mask=cpd_self_attn_mask)
        mol_emb = self.last_layer(cpd_enc_output)
        return mol_emb

    def target_base(self):
        target_emb = self.net(self.protein_go)
        return target_emb

    def disease_base(self):
        disease_emb = self.disease_net(self.disease_feature)
        return disease_emb

    def aggragate_emb(self, kg):
        ent_emb = self.init_embed
        rel_emb = self.init_rel
        for edge_layer, node_layer in zip(self.edge_layers, self.node_layers):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            ent_emb = ent_emb + node_ent_emb + edge_ent_emb
            ent_emb = self.bn(ent_emb)
        return ent_emb, rel_emb

    def emb_fusion(self,drug_emb_sub, target_emb, disease_emb, topology_emb, subj):

        drug_id = subj[subj < self.p.num_drugs]
        drug_sub = torch.index_select(topology_emb, 0, drug_id)
        drug_emb_sub = self.w1 * drug_sub + (1 - self.w1) * drug_emb_sub
        target_topology_emb = topology_emb[self.p.num_drugs:self.p.num_drugs+self.p.num_targets]
        target_emb = self.w2 * target_topology_emb + (1 - self.w2) * target_emb
        target_id = subj[subj >= self.p.num_drugs] - self.p.num_drugs
        target_emb_sub = target_emb[target_id]
        sub_emb = self.get_sub_emb(subj, drug_emb_sub, target_emb_sub)
        disease_emb = self.w3 * topology_emb[-self.p.num_diseases:] + (1 - self.w3) * disease_emb
        return sub_emb, target_emb, disease_emb


    def get_sub_emb(self,subj, drug_emb, target_emb):
        sub_emb = torch.empty((len(subj), self.init_dim)).cuda()
        drug_index = [i for i, j in enumerate(subj.tolist()) if j < self.p.num_drugs]
        target_index = [i for i, j in enumerate(subj.tolist()) if j >= self.p.num_drugs]
        sub_emb[drug_index, :] = drug_emb
        sub_emb[target_index, :] = target_emb
        return sub_emb


