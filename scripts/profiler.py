# profile.py
import torch
import esm
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.data import Data
from model import GilgameshPredictorV2
import data
import config

def load_ensemble():
    states = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    models = []
    dummy_dim = 26 
    for state in states:
        m = GilgameshPredictorV2(dummy_dim).to(config.DEVICE)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models

def profile_molecule(smiles):
    with open(config.RECEPTOR_SEQS_PATH, 'r') as f:
        seqs = json.load(f)
    
    print("Loading ESM for embedding...")
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval().to(config.DEVICE)
    
    ensemble = load_ensemble()
    
    # 2. Build Ligand Graph
    graph = data.get_mol_graph(smiles)
    if graph is None:
        print("Invalid SMILES")
        return

    # 3. Run Panel
    results = {}
    print(f"Profiling {smiles[:20]}...")
    
    for name, seq in seqs.items():
        # Embed Protein
        with torch.no_grad():
            _, _, tokens = batch_converter([("protein", seq)])
            tokens = tokens.to(config.DEVICE)
            res = esm_model(tokens, repr_layers=[12])
            p_emb = res['representations'][12][0, 1:-1].mean(0).unsqueeze(0).unsqueeze(0)
        
        # Prepare Batch
        batch = graph.clone().to(config.DEVICE)
        batch.protein_emb = p_emb
        batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long).to(config.DEVICE)
        
        # Predict (Ensemble Average)
        preds = []
        with torch.no_grad():
            for m in ensemble:
                preds.append(m(batch).item())
        avg_pred = sum(preds) / len(preds)
        results[name] = avg_pred

    # 4. Visualize
    visualize(results, smiles)

def visualize(scores, smiles):
    names = list(scores.keys())
    log_vals = list(scores.values())
    nm_vals = [10**x for x in log_vals]
    
    colors = []
    for n in log_vals:
        if n < 0: colors.append("red")
        else : colors.append("green")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=log_vals, hue=names, palette=colors, legend=False)
    plt.axhline(y=2, color='black', linestyle='--', label="Active Threshold (100 nM)")
    plt.ylabel("Predicted Log10 Ki (Lower = Stronger)")
    plt.title(f"Selectivity Profile\n{smiles[:30]}...")
    plt.legend()
    plt.show()

    print(f"{'Receptor':<10} | {'nM':<10} | {'Strength'}")
    print("-" * 35)
    for n, v in zip(names, nm_vals):
        s = "Extreme" if v < 10 else "Strong" if v < 100 else "Weak"
        print(f"{n:<10} | {v:<10.2f} | {s}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", type=str, help="SMILES string of the molecule")
    args = parser.parse_args()
    profile_molecule(args.smiles)