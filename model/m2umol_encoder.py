import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch.nn import Dropout, Linear, ModuleList, Sequential
from torch_geometric.nn import GINEConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch


class GPSConv1(nn.Module):
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(GPSConv1, self).__init__()
        self.channels = channels
        self.conv = conv
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True, **(attn_kwargs or {}))

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = "batch" in signature.parameters

    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, batch: Optional[torch.Tensor] = None, **kwargs):
        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                h = self.norm1(h, batch=batch) if self.norm_with_batch else self.norm1(h)
            hs.append(h)

        h, mask = to_dense_batch(x, batch)
        h, attweight = self.attn(
            h,
            h,
            h,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=True,
        )
        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        if self.norm2 is not None:
            h = self.norm2(h, batch=batch) if self.norm_with_batch else self.norm2(h)
        hs.append(h)

        out = sum(hs)
        out = out + self.mlp(out)
        if self.norm3 is not None:
            out = self.norm3(out, batch=batch) if self.norm_with_batch else self.norm3(out)
        return out, attweight


class M2UMolDrugEncoder(nn.Module):
    def __init__(self, dropout=0.3, attdropout=0.5):
        super(M2UMolDrugEncoder, self).__init__()
        channels = 128
        num_layers = 3

        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")
        self.encoder2D = ModuleList()
        attn_kwargs = {"dropout": attdropout}
        for _ in range(num_layers):
            nn1 = Sequential(
                Linear(channels, channels),
                nn.ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv1(
                channels,
                GINEConv(nn1),
                heads=4,
                attn_kwargs=attn_kwargs,
                dropout=dropout,
            )
            self.encoder2D.append(conv)

        self.pe_lin = Linear(20, 8)
        self.pe_norm = nn.BatchNorm1d(20)
        self.norm = nn.LayerNorm(128, 1e-5)
        self.linx = Linear(103, 120)
        self.linedgea = Linear(6, 128)
        self.linto3d = Linear(128, 128)
        self.lintotext = Linear(128, 128)
        self.lintobio = Linear(128, 128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.attnfea = nn.MultiheadAttention(128, 4, batch_first=True, dropout=attdropout)
        self.linfi = Linear(128, 128)
        self.normf = nn.BatchNorm1d(128)
        self.classification_mlp = Linear(128, 1)

    def forward(self, graph):
        graph = self.transform(graph)
        x_pe = self.pe_norm(graph.pe)
        x = torch.cat((self.linx(graph.x.float()), self.pe_lin(x_pe)), dim=1)
        edge_attr = self.linedgea(graph.edge_attr.float())

        for conv in self.encoder2D:
            x, _ = conv(x, graph.edge_index, graph.batch, edge_attr=edge_attr)

        feats2d = global_add_pool(x, graph.batch)
        z1 = self.norm(feats2d)

        z3d = self.norm(self.linto3d(feats2d)).unsqueeze(1)
        ztext = self.norm(self.lintotext(feats2d)).unsqueeze(1)
        zbio = self.norm(self.lintobio(feats2d)).unsqueeze(1)
        z1 = z1.unsqueeze(1)

        z_modal = torch.cat((z3d, ztext, zbio), dim=1)
        drug_emb, _ = self.attnfea(z1, z_modal, z_modal, average_attn_weights=True)
        drug_emb = z1 + drug_emb
        drug_emb = drug_emb.squeeze(1).view(-1, 128)
        drug_emb = self.normf(drug_emb)
        return drug_emb


def load_m2umol_pretrained(encoder, pretrained_path, map_location="cpu"):
    if not pretrained_path:
        return False, 0, 0

    model_dict = encoder.state_dict()
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=map_location, weights_only=False)
    except TypeError:
        pretrained_dict = torch.load(pretrained_path, map_location=map_location)
    if isinstance(pretrained_dict, dict) and "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]

    compatible = {}
    for name, value in pretrained_dict.items():
        clean_name = name
        for prefix in ("module.", "m2umol_encoder.", "m2u_encoder."):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        if clean_name in model_dict and model_dict[clean_name].shape == value.shape:
            compatible[clean_name] = value

    model_dict.update(compatible)
    encoder.load_state_dict(model_dict)
    return True, len(compatible), len(pretrained_dict)
