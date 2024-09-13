#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''==============================================

================================================'''

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Embeddings(nn.Module):
    def __init__(self, d_feature, d_model, dropout):
        super(Embeddings, self).__init__()
        self.ln = nn.Linear(d_feature, d_model)
        self.activation_fun = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.activation_fun(self.ln(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len, d_model]
    seq_k: [batch_size, seq_len, d_model]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    pad_attn_mask = torch.sum(torch.abs(seq_k), dim=-1) == 0  # [bitch_size, len_k]
    pad_attn_mask = pad_attn_mask.unsqueeze(1)
    attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    return attn_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, enc_dec_flag, dropout=0.1, activation_fun='softmax'):
        super(ScaledDotProductAttention, self).__init__()
        self.enc_dec_flag = enc_dec_flag
        self.d_k = d_k
        if self.enc_dec_flag:
            self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        if activation_fun == 'softmax':
            self.activation_fun = nn.Softmax(dim=-1)
        elif activation_fun == 'relu':
            self.activation_fun = nn.ReLU()
        elif activation_fun == 'sigmoid':
            self.activation_fun = nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, attn_mask_col, adj_matrix=None, dist_matrix=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        adj_matrix,dist_matrix,attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = scores.masked_fill(attn_mask_col, -1e9)
        scores = self.activation_fun(scores)

        if self.enc_dec_flag:
            batch_size, n_heads, len_q, len_k = scores.shape
            scores = scores.reshape(-1, len_q, len_k)
            dist_matrix = dist_matrix.reshape(-1, len_q, len_k)
            adj_matrix = adj_matrix.reshape(-1, len_q, len_k)
            con_matrix = torch.stack([scores, dist_matrix, adj_matrix], dim=1)
            weighted_sores = self.conv(con_matrix)
            weighted_sores = weighted_sores.squeeze(1)
            weighted_sores = weighted_sores.reshape(batch_size, n_heads, len_q, len_k)
            weighted_sores = weighted_sores.masked_fill(attn_mask_col, -1e9)
            weighted_sores = self.activation_fun(weighted_sores)
        else:
            weighted_sores = scores
        attn = self.dropout(weighted_sores)
        context = torch.matmul(attn, V)

        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, enc_dec_flag, distance_matrix_kernel='exp', activation_fun='softmax'):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0
        self.enc_dec_flag = enc_dec_flag
        self.d_v = self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, self.d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        if self.enc_dec_flag:
            self.dot_product_atten = ScaledDotProductAttention(self.d_k, self.enc_dec_flag, dropout, activation_fun)

            if distance_matrix_kernel == 'softmax':
                self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
            elif distance_matrix_kernel == 'exp':
                self.distance_matrix_kernel = lambda x: torch.exp(-x)
            elif distance_matrix_kernel == 'sigmoid':
                self.distance_matrix_kernel = lambda x: torch.sigmoid(-x)
        else:
            self.dot_product_atten = ScaledDotProductAttention(d_k=self.d_k, enc_dec_flag=self.enc_dec_flag,
                                                               dropout=dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask, adj_matrix=None, dist_matrix=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k] [4, 8, 50, 16]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        dist_matrix = dist_matrix.masked_fill(attn_mask, np.inf)
        dist_matrix = self.distance_matrix_kernel(dist_matrix)  # 补加行没有被mask,在attension模块中与其他特征一起处理
        dist_matrix = dist_matrix.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # [batch_size, n_heads, seq_len, seq_len]

        adj_matrix = adj_matrix.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                    1)  # [batch_size, n_heads, seq_len, seq_len]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # [batch_size, n_heads, seq_len, seq_len]

        context, attn = self.dot_product_atten(Q, K, V, attn_mask, adj_matrix, dist_matrix)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_v)  # context: [batch_size, len_model, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_v, d_model]
        return output, attn  # ,同时，ffn的正则化提到开始，也做相同的残差连接。

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ffn, d_model, bias=False)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        output = self.fc(inputs)
        return output  # [batch_size, seq_len, d_model]

class Encoder(nn.Module):
    def __init__(self, cpd_atom, d_model=128, n_heads=8, dropout=0.1,
                 distance_matrix_kernel='softmax',
                 d_ffn=256, activation_fun='softmax'):
        super(Encoder, self).__init__()

        self.comp_emb = Embeddings(cpd_atom, d_model, dropout)  # 这个是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.comp_pos_emb = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout,
                                                enc_dec_flag=True,
                                                distance_matrix_kernel=distance_matrix_kernel,
                                                activation_fun=activation_fun)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ffn, dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, cpd_self_attn_mask):
        cpd_enc_output = self.comp_emb(cpd_atom_features)  # [batch_size, 90, d_model]
        cpd_enc_output = self.comp_pos_emb(cpd_enc_output.transpose(0, 1)).transpose(0, 1)
        cpd_enc_attn_list = []
        residual = cpd_enc_output
        enc_inputs = self.norm(cpd_enc_output)
        enc_outputs, attn = self.self_attn(enc_inputs, enc_inputs, enc_inputs, cpd_self_attn_mask, cpd_adj_matrix,
                                               cpd_dist_matrix)  # enc_inputs to same Q,K,V
        residual = residual + self.dropout(enc_outputs)
        ffn_inputs = self.norm(residual)
        ffn_outputs = self.pos_ffn(ffn_inputs)
        cpd_enc_output = residual + self.dropout(ffn_outputs)
        cpd_enc_attn_list.append(attn)
        cpd_enc_output = self.layer_norm(cpd_enc_output)
        return cpd_enc_output,  cpd_enc_attn_list

class last_layer(nn.Module):
    def __init__(self, d_model, dropout=0.1, n_output=1):
        super(last_layer, self).__init__()

        self.cpd_dropout = nn.Dropout(p=dropout)
        self.prt_dropout = nn.Dropout(p=dropout)

        self.W1 = nn.Linear(d_model, 1, bias=False)
        self.W2 = nn.Linear(d_model, 1, bias=False)

        self.activation_fun = nn.ReLU()

        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

        self.ln = nn.Linear(2 * d_model, n_output, bias=False)

    def forward(self, cpd_atom_feats):
        batch_size = cpd_atom_feats.size(0)

        W1_output = self.cpd_dropout(self.activation_fun(self.W1(cpd_atom_feats)))
        W1_output = W1_output.view(batch_size, -1)
        W1_output = self.softmax1(W1_output)
        cf = torch.sum(cpd_atom_feats * W1_output.view(batch_size, -1, 1), dim=1)

        return cf
