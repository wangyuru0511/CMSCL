import os
import numpy as np
import dgl
import tqdm
import torch
import pickle
from sklearn.model_selection import train_test_split
from utils.KFold import KFold
import pandas as pd
from collections import Counter



def load_data(data_path='./data', device='cpu'):
    if os.path.exists(os.path.join(data_path, 'kg_data.pkl')):
        with open(os.path.join(data_path, 'kg_data.pkl'), 'rb') as save_data:
            kg_g, e_feat = pickle.load(save_data)
    else:
        # g
        edges = []
        e_feat = []
        with open(os.path.join(data_path, 'edges.tsv'), 'r') as file:
            for line in tqdm.tqdm(file.readlines(), desc='loading kg'):
                h, r, t = line.strip().split('\t')
                [h, r, t] = [int(h), int(r), int(t)]
                if h == t:
                    continue
                edges.append([h, t])
                edges.append([t, h])
                e_feat.append(r + 1)
                e_feat.append(r + 1)
                # edge_type[(h, t)] = r + 1
                # edge_type[(t, h)] = r + 1

        kg_g = dgl.graph(edges)
        kg_g = dgl.remove_self_loop(kg_g)  # 消除自环
        kg_g = dgl.add_self_loop(kg_g)  # 给孤立的点加自环

        e_feat = np.concatenate([np.array(e_feat), np.zeros(kg_g.num_edges() - len(e_feat), ).astype(np.int64)])

        nodes_list = []
        with open(os.path.join(data_path, 'nodes.tsv'), 'r') as file:
            for line in file.readlines():
                i, node_name, node_type = line.strip().split('\t')
                nodes_list.append([int(i), int(node_type)])
                node_ids = [node[0] for node in nodes_list]

        kg_g.ndata.update({'nodes': torch.tensor(nodes_list)})
        kg_g.edata.update({'edges': torch.from_numpy(e_feat)})

        with open(os.path.join(data_path, 'kg_data.pkl'), 'wb') as file:
            pickle.dump([kg_g, e_feat], file)

    smiles_list = []
    with open(os.path.join(data_path, 'smiles.tsv'), 'r') as file:
        for line in file.readlines():
            smiles = line.strip().split('\t')[1]
            smiles_list.append(smiles)

    return kg_g.to(device), smiles_list


def get_train_test(data_path='./data', fold_num=5, condition='s1'):
    sample = pd.read_csv(os.path.join(data_path, 'ddi.tsv'), sep='\t').values

    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42, condition=condition)

    final_train_samples = []
    final_valid_samples = []
    final_test_samples = []

    print(f"Start spliting data (Condition: {condition})...")

    for fold_idx, (train_val_data, test_data) in enumerate(kfold.split(sample)):  # kfold.split(sample)生成训练和测试数据，这个循环生成5-折交叉验证的训练、测试数据
        y = train_val_data[:, 2]
        counts = Counter(y)
        invalid_classes = [cls for cls, count in counts.items() if count < 2]

        if invalid_classes:
            mask = np.array([label in invalid_classes for label in y])
            rare_data = train_val_data[mask]
            train_val_data = train_val_data[~mask]
            y = train_val_data[:, 2]
        else:
            rare_data = None

        real_train_data, valid_data = train_test_split(
            train_val_data,
            test_size=0.125,
            random_state=42,
            shuffle=True,
            stratify=y
        )

        if rare_data is not None and len(rare_data) > 0:
            real_train_data = np.concatenate([real_train_data, rare_data])
            
        final_train_samples.append(real_train_data)
        final_valid_samples.append(valid_data)
        final_test_samples.append(test_data)

        print(f"Fold {fold_idx}: Train {len(real_train_data)} | Valid {len(valid_data)} | Test {len(test_data)}")
    print("Data spliting completed")

    return final_train_samples, final_valid_samples, final_test_samples
