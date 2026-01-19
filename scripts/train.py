# train.py
import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
import config
import data
from model import GilgameshPredictorV2
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.metrics import mean_squared_error, r2_score

def train_ensemble(num_models=3):
    raw_data = data.prepare_dataset()
    train_raw, test_raw = data.scaffold_split(raw_data)
    
    train_loader = GeoDataLoader(data.DrugDataset(train_raw), batch_size=config.HYPERPARAMS['batch_size'], shuffle=True)
    test_loader = GeoDataLoader(data.DrugDataset(test_raw), batch_size=config.HYPERPARAMS['batch_size'])
    
    num_features = train_raw[0][0].x.shape[1]
    saved_models = []

    for i in range(num_models):
        print(f"\nTraining Model {i+1}/{num_models}...")
        model = GilgameshPredictorV2(num_features).to(config.DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=config.HYPERPARAMS['learning_rate'], weight_decay=1e-3)
        crit = nn.MSELoss()
        
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(config.HYPERPARAMS['epochs']):
            model.train()
            loss_sum = 0
            for batch in tqdm(train_loader, leave=False):
                batch = batch.to(config.DEVICE)
                opt.zero_grad()
                pred = model(batch)
                loss = crit(pred, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()
            
            # Val
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(config.DEVICE)
                    preds.extend(model(batch).cpu().numpy())
                    trues.extend(batch.y.cpu().numpy())
            
            rmse = np.sqrt(mean_squared_error(trues, preds))
            if rmse < best_loss:
                best_loss = rmse
                best_state = copy.deepcopy(model.state_dict())
                
        print(f"Model {i+1} Best RMSE: {best_loss:.4f}")
        saved_models.append(best_state)

    torch.save(saved_models, config.MODEL_SAVE_PATH)
    print("Ensemble Saved!")

if __name__ == "__main__":
    train_ensemble()