# config.py
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEROTONIN_RECEPTORS = {
    "5-HT1A": "P08908",
    "5-HT2A": "P28223",
    "5-HT2C": "P28335",
    "5-HT3A": "P46098",
    "5-HT4":  "P31645",
    "5-HT5A": "P41437", # Doesn't exist in BindingDB
    "5-HT6":  "P50406",
    "5-HT7":  "P34969"
}

HYPERPARAMS = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-3,
    "hidden_dim": 128,
    "dropout": 0.4,
    "protein_emb_dim": 480,
    "epochs": 30
}

DATA_PATH = "BindingDB_All.tsv"
MODEL_SAVE_PATH = "model_ensemble.pt"
RECEPTOR_SEQS_PATH = "receptor_sequences.json"