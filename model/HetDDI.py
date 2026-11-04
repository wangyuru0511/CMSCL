import torch
import torch.nn.functional as F
import torch.nn as nn
from model.hgnn import HGNN
from model.decoder import Mlp


class HetDDI(nn.Module):
    def __init__(self,
                 kg_g,
                 smiles,
                 hidden_dim,
                 num_layer,
                 mode,
                 class_num,
                 condition,
                 freeze_m2umol=False,
                 proj_dim=128,
                 proj_dropout=0.2,
                 ):
        super(HetDDI, self).__init__()

        self.smiles = smiles
        self.device = kg_g.device
        self.mode = mode
        self.drug_num = len(smiles)

        dropout = 0.1

        if self.mode == 'kg' or self.mode == 'concat':
            self.kg = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'], hidden_dim, num_layer)
            self.kg_size = self.kg.get_output_size()
            self.kg_fc = nn.Sequential(nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU(),

                                       nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU(),

                                       nn.Linear(self.kg_size, self.kg_size),
                                       nn.BatchNorm1d(self.kg_size),
                                       nn.Dropout(dropout),
                                       nn.ReLU()
                                       )

        if self.mode == 'use_m2umol_online' or self.mode == 'concat':
            from model.m2umol_encoder import M2UMolDrugEncoder

            self.m2umol_encoder = M2UMolDrugEncoder(dropout=0.3, attdropout=0.5)
            if freeze_m2umol:
                for p in self.m2umol_encoder.parameters():
                    p.requires_grad = False
            self.m2u_proj = nn.Linear(128, hidden_dim)
            self.m2u_norm = nn.LayerNorm(hidden_dim)
            self.mol_size = hidden_dim

        if self.mode == 'kg':
            pair_dim = self.kg_size * 2
            self.decoder = Mlp(self.kg_size, 0, class_num=class_num)
        elif self.mode == 'use_m2umol_online':
            pair_dim = self.mol_size * 2
            self.decoder = Mlp(0, self.mol_size, class_num=class_num)
        elif self.mode == 'concat':
            pair_dim = (self.kg_size + self.mol_size) * 2
            self.decoder = Mlp(self.kg_size, self.mol_size, class_num=class_num)
        else:
            raise ValueError(f'Unsupported mode: {self.mode}')

        self.scl_proj = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Accept projection weights saved before the SCL naming update.
        legacy_prefix = prefix + 'sbcl_proj.'
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                new_key = prefix + 'scl_proj.' + key[len(legacy_prefix):]
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def scl_project(self, pair_repr):
        z = self.scl_proj(pair_repr)
        return F.normalize(z, dim=-1)

    def encode_m2umol(self, graph_batch):
        if graph_batch is None:
            raise ValueError('M2UMol online mode requires PyG graph batches.')
        graph_batch = graph_batch.to(self.device)
        mol_emb = self.m2umol_encoder(graph_batch)
        mol_emb = self.m2u_norm(self.m2u_proj(mol_emb))
        return mol_emb

    def forward(
            self,
            left,
            right,
            left_m2u_graph=None,
            right_m2u_graph=None,
            return_repr=False,
            return_pair_emb=False,
    ):
        need_pair_emb = return_repr or return_pair_emb
        if self.mode == 'kg':
            kg_emb = self.kg()[:self.drug_num]
            kg_emb = self.kg_fc(kg_emb)

            left_kg_emb = kg_emb[left]
            right_kg_emb = kg_emb[right]

            pair_repr = torch.concat([left_kg_emb, right_kg_emb], dim=-1)
            logits = self.decoder(left_kg_emb, right_kg_emb)
            if need_pair_emb:
                return logits, pair_repr
            return logits

        elif self.mode == 'use_m2umol_online':
            left_mol_emb = self.encode_m2umol(left_m2u_graph)
            right_mol_emb = self.encode_m2umol(right_m2u_graph)

            pair_repr = torch.concat([left_mol_emb, right_mol_emb], dim=-1)
            logits = self.decoder(left_mol_emb, right_mol_emb)
            if need_pair_emb:
                return logits, pair_repr
            return logits

        elif self.mode == 'concat':
            kg_emb = self.kg()[:self.drug_num]
            kg_emb = self.kg_fc(kg_emb)

            left_kg_emb = kg_emb[left]
            right_kg_emb = kg_emb[right]

            left_mol_emb = self.encode_m2umol(left_m2u_graph)
            right_mol_emb = self.encode_m2umol(right_m2u_graph)

            left_emb = torch.concat([left_kg_emb, left_mol_emb], dim=-1)
            right_emb = torch.concat([right_kg_emb, right_mol_emb], dim=-1)

            pair_repr = torch.concat([left_emb, right_emb], dim=-1)
            logits = self.decoder(left_emb, right_emb)
            if need_pair_emb:
                return logits, pair_repr
            return logits

        raise ValueError(f'Unsupported mode: {self.mode}')
