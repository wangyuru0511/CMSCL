import argparse
import os

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    try:
        implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
    except AttributeError:
        implicit_valence = atom.GetImplicitValence()

    results = (
        one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ["Unknown"])
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(implicit_valence, [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [atom.GetIsAromatic()]
    )
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"]) + [
                atom.HasProp("_ChiralityPossible")
            ]
        except Exception:
            results = results + [False, False] + [atom.HasProp("_ChiralityPossible")]
    return torch.tensor(np.array(results).astype(np.float32))


def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor(
        [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ],
        dtype=torch.float,
    )


def mol_to_m2u_graph(drug_id, smiles, atom_symbols):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol.GetAtoms()]
    features.sort()
    if not features:
        return None
    _, features = zip(*features)
    x = torch.stack(features).float()

    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        attr = edge_features(bond)
        edges.append([begin, end])
        edges.append([end, begin])
        edge_attrs.append(attr)
        edge_attrs.append(attr)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs).float()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    if x.size(1) != 103:
        raise ValueError(f"Expected atom feature dim 103, got {x.size(1)} for drug {drug_id}")

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, drug_id=int(drug_id), smiles=smiles)


def read_smiles(data_dir):
    rows = []
    with open(os.path.join(data_dir, "smiles.tsv"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append((int(parts[0]), parts[1]))
    return rows


def build_cache(data_dir, symbo_path):
    atom_symbols = np.load(symbo_path, allow_pickle=True).tolist()
    graphs = []
    drug_id_to_idx = {}
    failed_drugs = []

    for drug_id, smiles in read_smiles(data_dir):
        try:
            graph = mol_to_m2u_graph(drug_id, smiles, atom_symbols)
        except Exception as exc:
            failed_drugs.append((drug_id, str(exc)))
            continue
        if graph is None:
            failed_drugs.append((drug_id, "RDKit failed to parse SMILES"))
            continue
        drug_id_to_idx[drug_id] = len(graphs)
        graphs.append(graph)

    return {"drug_id_to_idx": drug_id_to_idx, "graphs": graphs, "failed_drugs": failed_drugs}


def main():
    parser = argparse.ArgumentParser(description="Build M2UMol PyG graph cache for CMSCL drugs.")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--kg_name", type=str, default="DRKG")
    parser.add_argument("--ddi_names", nargs="+", default=["Deng", "Ryu"], choices=["Deng", "Ryu"])
    parser.add_argument(
        "--symbo_path",
        type=str,
        default="./data/symbo.npy",
    )
    args = parser.parse_args()
    if not os.path.exists(args.symbo_path):
        raise FileNotFoundError(
            f"Cannot find symbo.npy at {args.symbo_path}. "
            "Pass --symbo_path to the copied M2UMol symbo.npy file."
        )

    for ddi_name in args.ddi_names:
        data_dir = os.path.join(args.data_path, f"{args.kg_name}+{ddi_name}")
        cache = build_cache(data_dir, args.symbo_path)
        output_path = os.path.join(data_dir, "m2umol_graph_cache.pt")
        torch.save(cache, output_path)
        print(f"{ddi_name}: built M2UMol graphs: {len(cache['graphs'])}")
        print(f"{ddi_name}: failed drugs: {len(cache['failed_drugs'])}")
        print(f"{ddi_name}: saved graph cache to {output_path}")


if __name__ == "__main__":
    main()
