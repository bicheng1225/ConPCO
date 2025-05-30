# -*- coding: utf-8 -*-
# @Author  : Bi-Cheng Yan
# @Affiliation  : National Taiwan Normal University
# @Email   : bicheng@ntnu.edu.tw
# @File    : gopt_ssl_3m_bfr_cat_utt_clap.py

# attention part is borrowed from the timm package.

import math
import warnings

import torch
import torch.nn as nn
import numpy as np

# from espnet.nets.pytorch_backend.nets_utils import pad_list
def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

# code from the t2t-vit paper
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #print(C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BlockCNN(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.size = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #[in_channel, out_channel, kernel_size]
        self.conv1d = nn.Conv1d(dim, dim*2, kernel_size=1)
        self.glu_act = nn.GLU()
        self.cnn_norm1 = norm_layer(dim)
        self.cnn_norm2 = norm_layer(dim)
        #depth-wise conv
        self.dep_cnn1 = nn.Conv1d(dim, dim, kernel_size=3, groups=dim, padding=1)
        self.dep_cnn2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.cnn_drop = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

        # attention-based pooling for two branches
        self.pooling_proj1 = torch.nn.Linear(dim, 1)
        self.pooling_proj2 = torch.nn.Linear(dim, 1)
        # linear projections for calculating merging weights
        self.weight_proj1 = torch.nn.Linear(dim, 1)
        self.weight_proj2 = torch.nn.Linear(dim, 1)
        # linear projection after weighted average
        self.merge_proj = torch.nn.Linear(dim, dim)

        self.mlp2 = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # self atten block
        x_att = x + self.drop_path(self.attn(self.norm1(x)))
        x_att = x_att + self.drop_path(self.mlp(self.norm2(x_att)))
        # conv block, #x [B, 50, dim]
        x_cnn = self.conv1d(self.cnn_norm1(x).transpose(1,2)).transpose(1,2) #x_cnn [B, 50, dim*2]
        x_cnn = self.glu_act(x_cnn) #x_cnn [B, 50, dim]
        # depth-wise conv
        x_cnn = self.dep_cnn1(x_cnn.transpose(1,2))
        x_cnn = self.dep_cnn2(x_cnn).transpose(1,2)
        x_cnn = torch.nn.functional.relu(self.cnn_norm2(x_cnn))
        x_cnn = self.cnn_drop(self.conv1d_2(x_cnn.transpose(1,2)).transpose(1,2))
        x_cnn = x + x_cnn 

        # branch1 for atten_out attention pooling
        score1 = (
            self.pooling_proj1(x_att).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score1 = torch.softmax(score1, dim=-1)
        pooled1 = torch.matmul(score1, x_att).squeeze(1)  # (batch, size)
        weight1 = self.weight_proj1(pooled1)  # (batch, 1)

        # branch2 for cnn_out attention pooling
        score2 = (
            self.pooling_proj2(x_cnn).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score2 = torch.softmax(score2, dim=-1)
        pooled2 = torch.matmul(score2, x_cnn).squeeze(1)  # (batch, size)
        weight2 = self.weight_proj2(pooled2)  # (batch, 1)

        # normalize weights of two branches
        merge_weights = torch.softmax(
            torch.cat([weight1, weight2], dim=-1), dim=-1
        )  # (batch, 2)
        merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)
        w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

        # merge and proj
        x = x + self.dropout(
            self.mlp2(w1 * x_att + w2 * x_cnn)
        )
        
        return x

class W2UFeatGen(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.size = dim
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(dim)
        # attention-based pooling for two branches
        self.pooling_proj1 = torch.nn.Linear(dim, 1)
        self.pooling_proj2 = torch.nn.Linear(dim, 1)
        self.pooling_proj3 = torch.nn.Linear(dim, 1)
        # linear projections for calculating merging weights
        self.weight_proj1 = torch.nn.Linear(dim, 1)
        self.weight_proj2 = torch.nn.Linear(dim, 1)
        self.weight_proj3 = torch.nn.Linear(dim, 1)
        # linear projection after weighted average
        self.merge_proj = torch.nn.Linear(dim, dim)
        self.mlp2 = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.w1_proj = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.w2_proj = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.w3_proj = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dropout = nn.Dropout(0.2)

    def forward(self, w1, w2, w3):
        w1_proj, w2_proj, w3_proj = self.w1_proj(w1), self.w2_proj(w2), self.w3_proj(w3)

        # branch1 for w1_proj attention pooling
        score1 = (
            self.pooling_proj1(w1_proj).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score1 = torch.softmax(score1, dim=-1)
        pooled1 = torch.matmul(score1, w1_proj).squeeze(1)  # (batch, size)
        weight1 = self.weight_proj1(pooled1)  # (batch, 1)

        # branch2 for w2_proj attention pooling
        score2 = (
            self.pooling_proj2(w2_proj).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score2 = torch.softmax(score2, dim=-1)
        pooled2 = torch.matmul(score2, w2_proj).squeeze(1)  # (batch, size)
        weight2 = self.weight_proj2(pooled2)  # (batch, 1)

        # branch3 for w3_proj attention pooling
        score3 = (
            self.pooling_proj2(w3_proj).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score3 = torch.softmax(score3, dim=-1)
        pooled3 = torch.matmul(score3, w3_proj).squeeze(1)  # (batch, size)
        weight3 = self.weight_proj3(pooled3)  # (batch, 1)

        # normalize weights of two branches
        merge_weights = torch.softmax(
            torch.cat([weight1, weight2, weight3], dim=-1), dim=-1
        )  # (batch, 2)
        merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)
        w1, w2, w3 = merge_weights[:, 0], merge_weights[:, 1], merge_weights[:, 2]  # (batch, 1, 1)
        # merge and proj
        x = self.dropout(
            self.mlp(w1 * w1_proj + w2 * w2_proj + w3 * w3_proj)
        )
        return x

class AttentionPooling(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pooling_proj = torch.nn.Linear(dim, 1)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.mlp(x)
        # branch1 for w1_proj attention pooling
        score = (
            self.pooling_proj(x).transpose(1, 2) / x.size(-1)**0.5
        )  # (batch, 1, time)
        score = torch.softmax(score, dim=-1)
        pooled = torch.matmul(score, x).squeeze(1)  # (batch, size)

        return pooled

# standard HierCB model proposed in the paper
class HierCB(nn.Module):
    def make_word_pos_mask(self, ys_pad):
        #0 is mask, 1 value
        B = ys_pad.size()[0]
        L = ys_pad.size()[1]
        ys_mask = torch.zeros(B,L,L)
        for i in range(B):
            for idx, pos in enumerate(ys_pad[i]):
                #-1 padding symbol
                if pos == -1: 
                    break
                ys_mask[i][idx] = (ys_pad[i]==pos).int() 
        return ys_mask

    def __init__(self, embed_dim, num_heads, p_depth, w_depth, u_depth, ssl_drop, input_dim=84):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Transformer encode blocks
        self.phn_blocks = nn.ModuleList([BlockCNN(dim=embed_dim, num_heads=num_heads) for i in range(p_depth)])
        self.word_blocks = nn.ModuleList([BlockCNN(dim=embed_dim, num_heads=num_heads) for i in range(w_depth)])
        self.utt_blocks = nn.ModuleList([BlockCNN(dim=embed_dim, num_heads=num_heads) for i in range(u_depth)])

        # sin pos embedding or learnable pos embedding, 50 sequence length
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, self.embed_dim))
        self.word_pos_embed = torch.nn.Embedding(50, embed_dim, padding_idx=0)

        trunc_normal_(self.pos_embed, std=.02)

        # for phone classification
        self.p_in_proj = nn.Linear(self.input_dim + 1024*3, embed_dim)
        self.w_in_proj = nn.Linear(self.input_dim + 1024*3, embed_dim)
        self.u_in_proj = nn.Linear(self.input_dim + 1024*3, embed_dim)
        self.ssl_drop = nn.Dropout(ssl_drop)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # for word classification, 1=accuracy, 2=stress, 3=total
        self.word_input_att = MultiHeadedAttention(num_heads, embed_dim, 0.1)
        self.word_input_att1 = MultiHeadedAttention(num_heads, embed_dim, 0.1)

        self.w_in_cat_proj = MLP(in_features=embed_dim*2, hidden_features=embed_dim,
        out_features=embed_dim, act_layer=nn.GELU, drop=0.1)
        self.w_proj_ln1 = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.w_proj_ln2 = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.w_proj_ln3 = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

        self.w_proj_cnn1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.w_proj_cnn2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.w3_proj = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # canonical phone projection, assume there are 40 phns
        self.phn_proj = nn.Linear(40, embed_dim)
        self.word_proj = nn.Linear(2607, embed_dim)

        # utterance level, 1=accuracy, 2=completeness, 3=fluency, 4=prosodic, 5=total score
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.utt_feat_ext = W2UFeatGen(embed_dim)

        self.u1_att_pooling = AttentionPooling(embed_dim)
        self.u2_att_pooling = AttentionPooling(embed_dim)
        self.u3_att_pooling = AttentionPooling(embed_dim)
        self.u4_att_pooling = AttentionPooling(embed_dim)
        self.u5_att_pooling = AttentionPooling(embed_dim)

        self.u_in_cat_proj = MLP(in_features=embed_dim*3, hidden_features=embed_dim,
        out_features=embed_dim, act_layer=nn.GELU, drop=0.1)

        self.u_in_proj1 = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.u_in_proj2 = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.u_in_proj3 = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

        self.u_proj_cnn1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.u_proj_cnn2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.u_proj_cnn3 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        
        self.phn_audio_proj = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.phn_text_proj = MLP(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

    # x shape in [batch_size, sequence_len, feat_dim]
    # phn in [batch_size, seq_len]
    def forward(self, x, x_eng, x_dur, x_ssl, phn, word_pos, word):

        # batch size
        B = x.shape[0]

        # phn_one_hot in shape [batch_size, seq_len, feat_dim]
        phn_one_hot = torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()

        # phn_embed in shape [batch_size, seq_len, embed_dim]
        phn_embed = self.phn_proj(phn_one_hot)

        # if the input dimension is different from the Transformer embedding dimension, project the input to same dim
        if self.embed_dim != self.input_dim:
            x = torch.cat([x, self.ssl_drop(x_ssl), x_dur, x_eng], dim=-1)
            p_x, w_x, u_x = self.p_in_proj(x), self.w_in_proj(x), self.u_in_proj(x)

        p_x = p_x + phn_embed + self.pos_embed

        # forward to the Transformer encoder
        p_tmp_feat = []
        for blk in self.phn_blocks:
            p_x = blk(p_x)
            p_tmp_feat.append(p_x)
        p = self.mlp_head_phn(p_x)

        # word score is propagated to phone-level, so word output is also at phone-level.
        # but different mlp heads are used, 1 = accuracy, 2 = stress, 3 = total
        word_one_hot = torch.nn.functional.one_hot(word.long()+1, num_classes=2605+2).float()

        # word_embed in shape [batch_size, seq_len, embed_dim]
        word_embed = self.word_proj(word_one_hot)
        
        phn2word_msk = self.make_word_pos_mask(word_pos)
        phn2word_msk = phn2word_msk.to(p.device)

        # phone-level output
        w_p_x = self.w_proj_cnn1(p_x.transpose(1,2)).transpose(1,2)
        w_x = self.w_proj_cnn2(w_x.transpose(1,2)).transpose(1,2)

        # prepare word_level input
        w_x_att = self.word_input_att(w_x, w_x, w_x, phn2word_msk)
        w_p_x_att = self.word_input_att1(w_p_x, w_p_x, w_p_x, phn2word_msk)
        x_word = self.w_in_cat_proj(torch.cat([w_x_att, w_p_x_att], dim=-1))

        # add word_pos, word_embed
        x_word = x_word + word_embed + self.word_pos_embed(word_pos.int()+1) + p_x

        for blk in self.word_blocks:
            x_word = blk(x_word)
        
        w1_proj = self.w_proj_ln1(x_word)
        w2_proj = self.w_proj_ln2(x_word)
        w3_proj = self.w_proj_ln3(x_word)

        w1 = self.mlp_head_word1(w1_proj)
        w2 = self.mlp_head_word2(w2_proj)
        w3 = self.mlp_head_word3(w3_proj)

        u_p_x = p_x
        u_w_feats = self.utt_feat_ext(w1_proj, w2_proj, w3_proj)

        # spanning from phone-level to word-level, then utterance-level
        u_p_feats = self.u_proj_cnn1(u_p_x.transpose(1,2)).transpose(1,2)
        u_w_feats = self.u_proj_cnn2(u_w_feats.transpose(1,2)).transpose(1,2)
        utt_feats = self.u_in_cat_proj(torch.cat([u_p_feats, u_w_feats, u_x], dim=-1)) + p_x

        for blk in self.utt_blocks:
            utt_feats = blk(utt_feats)

        u1_proj = self.u1_att_pooling(utt_feats)
        u2_proj = self.u2_att_pooling(utt_feats)
        u3_proj = self.u3_att_pooling(utt_feats)
        u4_proj = self.u4_att_pooling(utt_feats)
        u5_proj = self.u5_att_pooling(utt_feats)

        u1 = self.mlp_head_utt1(u1_proj)
        u2 = self.mlp_head_utt2(u2_proj)
        u3 = self.mlp_head_utt3(u3_proj)
        u4 = self.mlp_head_utt4(u4_proj)
        u5 = self.mlp_head_utt5(u5_proj)

        return u1, u2, u3, u4, u5, p, w1, w2, w3, self.phn_audio_proj(p_tmp_feat[0]), self.phn_text_proj(phn_embed + self.pos_embed)
