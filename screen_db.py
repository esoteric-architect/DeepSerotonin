import torch
import pandas as pd
import numpy as np
import esm
import json
import os
import gc
from tqdm import tqdm
from torch_geometric.data import Batch
from collections import defaultdict
import config
import data
from model import GilgameshPredictorV2

BATCH_SIZE = 64
TOP_K = 1000
RESULTS_DIR = "screening_results_full_matrix"

def load_ensemble():
    print(f"Loading Ensemble from {config.MODEL_SAVE_PATH}...")
    states = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    models = []
    input_dim = states[0]['conv1.nn.0.weight'].shape[1]
    for state in states:
        m = GilgameshPredictorV2(input_dim).to(config.DEVICE)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models

def precompute_receptor_embeddings(device):
    print("Pre-computing receptor embeddings...")
    with open(config.RECEPTOR_SEQS_PATH, 'r') as f:
        seqs = json.load(f)
    
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval().to(device)
    
    embeddings = {}
    with torch.no_grad():
        for name, seq in tqdm(seqs.items(), desc="Embedding"):
            _, _, tokens = batch_converter([("protein", seq)])
            tokens = tokens.to(device)
            res = esm_model(tokens, repr_layers=[12])
            emb = res['representations'][12][0, 1:-1].mean(0).unsqueeze(0)
            embeddings[name] = emb.cpu()
            
    del esm_model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return embeddings

def process_batch_against_all(models, graph_batch, receptor_embeddings, device):
    results = defaultdict(list)
    graph_batch = graph_batch.to(device)
    
    for rec_name, rec_emb in receptor_embeddings.items():
        rec_emb_flat = rec_emb.to(device).view(1, -1) 
        rec_emb_batch = rec_emb_flat.expand(graph_batch.num_graphs, -1)
        rec_emb_final = rec_emb_batch.unsqueeze(1)
        graph_batch.protein_emb = rec_emb_final
        
        preds = []
        with torch.no_grad():
            for model in models:
                preds.append(model(graph_batch))

        avg_preds = torch.stack(preds).mean(dim=0).cpu().numpy()
        results[rec_name] = avg_preds # Store numpy array directly
        
    return results

def screen_database():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = config.DEVICE
    
    models = load_ensemble()
    rec_embeddings = precompute_receptor_embeddings(device)
    receptor_names = list(rec_embeddings.keys())
    top_hits = {name: [] for name in receptor_names}
    
    seen_smiles = set()
    buffer_graphs = []
    buffer_smiles = []
    
    print(f"Streaming BindingDB from {config.DATA_PATH}...")
    chunk_iter = pd.read_csv(config.DATA_PATH, sep='\t', usecols=['Ligand SMILES'], 
                             chunksize=100000, low_memory=False)
    
    total_processed = 0
    
    for chunk in chunk_iter:
        unique_smiles = set(chunk['Ligand SMILES'].dropna().unique())
        new_smiles = list(unique_smiles - seen_smiles)
        seen_smiles.update(new_smiles)
        
        if not new_smiles: continue
        
        for smiles in tqdm(new_smiles, desc=f"Scanning Chunk (Total {total_processed})", leave=False):
            graph = data.get_mol_graph(smiles)
            if graph is None: continue
            
            buffer_graphs.append(graph)
            buffer_smiles.append(smiles)
            
            if len(buffer_graphs) >= BATCH_SIZE:
                batch = Batch.from_data_list(buffer_graphs)
                batch_scores_dict = process_batch_against_all(models, batch, rec_embeddings, device)
                for i in range(len(buffer_smiles)):
                    smi = buffer_smiles[i]
                    mol_profile = {rec: float(batch_scores_dict[rec][i]) for rec in receptor_names}
                    for rec in receptor_names:
                        score_for_this_rec = mol_profile[rec]
                        top_hits[rec].append((score_for_this_rec, mol_profile, smi))
                
                for rec_name in top_hits:
                    if len(top_hits[rec_name]) > TOP_K * 2:
                        top_hits[rec_name].sort(key=lambda x: x[0])
                        top_hits[rec_name] = top_hits[rec_name][:TOP_K]
                
                buffer_graphs = []
                buffer_smiles = []
        
        total_processed += len(new_smiles)
        
        if total_processed % 50000 < len(new_smiles):
            save_results_full(top_hits, receptor_names)

    # Final Save
    save_results_full(top_hits, receptor_names)
    print(f"Screening Complete! Scanned {total_processed} molecules.")

def save_results_full(top_hits, receptor_names):
    for rec_name, hits in top_hits.items():
        hits.sort(key=lambda x: x[0])
        final_hits = hits[:TOP_K]
        rows = []
        for main_score, profile_dict, smi in final_hits:
            row = {"SMILES": smi}
            for r in receptor_names:
                row[r] = 10 ** profile_dict[r]
            rows.append(row)
        
        df = pd.DataFrame(rows)
        cols = ["SMILES", rec_name] + [r for r in receptor_names if r != rec_name]
        df = df[cols]
        
        safe_name = rec_name.replace("-", "").replace(" ", "")
        path = os.path.join(RESULTS_DIR, f"top_hits_{safe_name}.csv")
        df.to_csv(path, index=False)

if __name__ == "__main__":
    screen_database()