import torch
import torch.nn as nn
import dgl
import dgl.function as fn


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class NodeLayer(nn.Module):
    def __init__(self, h_dim, params,n_ent,n_rel):
        super().__init__()
        self.p = params
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.neigh_w = get_param((h_dim, h_dim))
        self.act = nn.Tanh()
        self.bn = torch.nn.BatchNorm1d(h_dim)

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']
            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)
            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)
            neigh_ent_emb = self.act(neigh_ent_emb)
        return neigh_ent_emb


class EdgeLayer(nn.Module):
    def __init__(self, h_dim, params,n_ent,n_rel):
        super().__init__()
        self.p = params
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.neigh_w = get_param((h_dim, h_dim))
        self.act = nn.Tanh()
        self.bn = torch.nn.BatchNorm1d(h_dim)


    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']
            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)
            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)
            neigh_ent_emb = self.act(neigh_ent_emb)
        return neigh_ent_emb
