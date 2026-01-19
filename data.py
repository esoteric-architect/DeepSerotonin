
import ssl
# ssl error for macos 
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import numpy as np
import torch
import esm
import json
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm
from collections import defaultdict
import config

def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def get_atom_features(atom):
    atom_type = one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B'])
    degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    hybridization = one_hot_encoding(atom.GetHybridization(), 
        [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, 
         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    
    return atom_type + degree + hybridization + \
           [1 if atom.GetIsAromatic() else 0] + \
           [1 if atom.IsInRing() else 0] + \
           [atom.GetFormalCharge()]

def get_mol_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        edge_index += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()], 
                       [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def prepare_dataset(path=config.DATA_PATH):
    print(f"Loading data from {path}...")
    cols = ["Ligand SMILES", 'UniProt (SwissProt) Primary ID of Target Chain',
            'Ki (nM)', 'BindingDB Target Chain Sequence']
    df = pd.read_csv(path, sep='\t', usecols=cols, low_memory=False)
    
    receptor_ids = list(config.SEROTONIN_RECEPTORS.values())
    df = df[
        df['UniProt (SwissProt) Primary ID of Target Chain'].isin(receptor_ids) &
        df['Ki (nM)'].notna() &
        df['BindingDB Target Chain Sequence'].notna() &
        df['Ligand SMILES'].notna()
    ]
    
    save_receptor_sequences(df)

    print("Loading ESM model for embedding...")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(config.DEVICE)

    valid_pairs = []
    memo = {}
    id_to_name = {v: k for k, v in config.SEROTONIN_RECEPTORS.items()}

    print("Generating embeddings...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['BindingDB Target Chain Sequence']
        smiles = row['Ligand SMILES']
        uniprot = row['UniProt (SwissProt) Primary ID of Target Chain']
        
        if len(seq) > 10000: continue
            
        graph = get_mol_graph(smiles)
        if graph is None: continue

        if seq not in memo:
            with torch.no_grad():
                _, _, tokens = batch_converter([("protein", seq)])
                tokens = tokens.to(config.DEVICE)
                res = model(tokens, repr_layers=[12], return_contacts=False)
                memo[seq] = res['representations'][12][0, 1:-1].mean(0).cpu().numpy()
        
        try:
            val = float(str(row['Ki (nM)']).replace('>','').replace('<',''))
            label = np.log10(val)
        except: continue

        valid_pairs.append((graph, memo[seq], label, smiles, id_to_name.get(uniprot, "Unknown")))

    return valid_pairs

def save_receptor_sequences(df):
    """Extracts one sequence per receptor and saves to JSON"""
    sequences = {}
    for name, uid in config.SEROTONIN_RECEPTORS.items():
        subset = df[df['UniProt (SwissProt) Primary ID of Target Chain'] == uid]
        if not subset.empty:
            sequences[name] = subset.iloc[0]['BindingDB Target Chain Sequence']
    
    with open(config.RECEPTOR_SEQS_PATH, 'w') as f:
        json.dump(sequences, f)
    print(f"Saved receptor sequences to {config.RECEPTOR_SEQS_PATH}")

def scaffold_split(data, test_size=0.2):
    scaffolds = defaultdict(list)
    for item in data:
        mol = Chem.MolFromSmiles(item[3])
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[s].append(item)
    
    keys = sorted(scaffolds.keys(), key=lambda x: len(scaffolds[x]), reverse=True)
    train, test = [], []
    cutoff = len(data) * (1 - test_size)
    
    for k in keys:
        if len(train) + len(scaffolds[k]) < cutoff:
            train.extend(scaffolds[k])
        else:
            test.extend(scaffolds[k])
            
    return train, test

class DrugDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.data = raw_data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        g, p, y, s, r = self.data[idx]
        g.protein_emb = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        g.y = torch.tensor([y], dtype=torch.float32)
        g.smiles = s
        g.receptor = r
        return g