# src/train.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# PyTorch and PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

warnings.filterwarnings("ignore")

# ===================================================================
# 1. 配置, 类定义与核心函数
# ===================================================================
print("--- [步骤 1/4] 加载配置，定义模型和辅助函数 ---")

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

os.makedirs(SETTINGS['MODEL_DIR'], exist_ok=True)

class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    HIDDEN_DIM, NUM_LAYERS, GAT_HEADS, DROPOUT = 384, 6, 8, 0.2
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES = 7, 2
    EPOCHS, BATCH_SIZE, LEARNING_RATE, ES_PATIENCE = 600, 64, 1e-4, 40
    N_FOLDS, SEED = 5, 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {CFG.DEVICE}")

def calculate_weighted_mae_official(y_true_df, y_pred_df, target_columns, train_stats_df):
    K = len(target_columns)
    sample_counts = {t: len(train_stats_df[t].dropna()) for t in target_columns}
    value_ranges = {t: train_stats_df[t].max() - train_stats_df[t].min() for t in target_columns}
    sqrt_inv_n = {t: np.sqrt(1.0 / sample_counts[t]) if sample_counts[t] > 0 else 0 for t in target_columns}
    sum_sqrt_inv_n = sum(sqrt_inv_n.values())
    if sum_sqrt_inv_n == 0: return 0.0, {}, {}
    weights = {t: (1.0 / value_ranges[t]) * (K * sqrt_inv_n[t]) / sum_sqrt_inv_n if value_ranges.get(t, 0) > 0 and value_ranges[t] > 0 else 0 for t in target_columns}
    weights_array = np.array([weights.get(t, 0) for t in target_columns])
    abs_errors = (y_pred_df[target_columns] - y_true_df[target_columns]).abs()
    weighted_errors = abs_errors * weights_array
    total_weighted_error_sum = weighted_errors.sum().sum()
    if len(y_true_df) == 0: return 0.0, {}, {}
    return total_weighted_error_sum / len(y_true_df), weights, {}

class WeightedMAELoss(nn.Module):
    def __init__(self, weights_dict, target_columns, device):
        super().__init__()
        weights_tensor = torch.tensor([weights_dict.get(t, 0) for t in target_columns], dtype=torch.float, device=device).unsqueeze(0)
        self.register_buffer('weights', weights_tensor)
    def forward(self, predictions, targets):
        mask = ~torch.isnan(targets)
        if not torch.any(mask): return torch.tensor(0.0, device=predictions.device)
        abs_error = torch.abs(predictions - targets)
        weighted_abs_error = abs_error * self.weights
        total_weighted_error_sum = torch.sum(weighted_abs_error[mask])
        return total_weighted_error_sum / predictions.shape[0]

class PolymerDataset(InMemoryDataset):
    def __init__(self, filepath):
        super().__init__()
        self.load(filepath)

class PolymerGNN_v12_Res(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_layers, tasks_fp_indices, targets, heads=4, dropout=0.2):
        super(PolymerGNN_v12_Res, self).__init__()
        assert hidden_dim % heads == 0
        head_dim = hidden_dim // heads
        self.tasks_fp_indices, self.targets, self.dropout = tasks_fp_indices, targets, dropout
        self.input_conv = GATv2Conv(num_node_features, head_dim, heads=heads, dropout=self.dropout, edge_dim=num_edge_features)
        self.input_bn = BatchNorm(hidden_dim)
        self.hidden_convs = nn.ModuleList([GATv2Conv(hidden_dim, head_dim, heads=heads, dropout=self.dropout, edge_dim=num_edge_features) for _ in range(num_layers - 1)])
        self.hidden_bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - 1)])
        self.task_predictors = nn.ModuleDict()
        for task_name in self.targets:
            if task_name in self.tasks_fp_indices and self.tasks_fp_indices[task_name]:
                k = len(self.tasks_fp_indices[task_name])
                self.task_predictors[task_name] = nn.Sequential(nn.Linear(hidden_dim + k, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))
    def forward(self, data):
        x, edge_index, edge_attr, batch, morgan_fp = data.x, data.edge_index, data.edge_attr, data.batch, data.morgan_fp
        x = F.elu(self.input_bn(self.input_conv(x, edge_index, edge_attr=edge_attr)))
        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            x = F.elu(bn(conv(x, edge_index, edge_attr=edge_attr))) + x
        graph_embedding = global_mean_pool(x, batch)
        outputs = []
        for task_name in self.targets:
            if task_name in self.task_predictors:
                indices = torch.tensor(self.tasks_fp_indices[task_name], dtype=torch.long, device=morgan_fp.device)
                selected_fp = morgan_fp.index_select(1, indices)
                fused_embedding = torch.cat([graph_embedding, selected_fp], dim=1)
                outputs.append(self.task_predictors[task_name](fused_embedding))
            else: outputs.append(torch.full((graph_embedding.size(0), 1), 0.0, device=graph_embedding.device))
        return torch.cat(outputs, dim=1)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train(); total_loss = 0
    for batch in loader:
        batch = batch.to(device); optimizer.zero_grad()
        predictions = model(batch); loss = loss_fn(predictions, batch.y)
        if not torch.isnan(loss) and loss > 0: loss.backward(); optimizer.step()
        if not torch.isnan(loss): total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0

def validate_and_get_preds(model, loader, device):
    model.eval(); all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device); predictions = model(batch)
            all_preds.append(predictions.cpu().numpy()); all_trues.append(batch.y.cpu().numpy())
    return np.concatenate(all_preds, axis=0) if all_preds else np.array([]), np.concatenate(all_trues, axis=0) if all_trues else np.array([])

# ===================================================================
# 2. 数据加载与wMAE权重预计算
# ===================================================================
print("\n--- [步骤 2/4] 加载处理好的数据并计算wMAE权重 ---")
full_dataset = PolymerDataset(filepath=SETTINGS['PROCESSED_TRAIN_DATA_PATH'])
all_y = np.array([data.y.numpy().flatten() for data in full_dataset])
full_train_df_for_stats = pd.DataFrame(all_y, columns=CFG.TARGETS)

_, weights_dict, _ = calculate_weighted_mae_official(full_train_df_for_stats, full_train_df_for_stats, CFG.TARGETS, full_train_df_for_stats)
print("wMAE权重计算完成:", {k: f"{v:.4f}" for k, v in weights_dict.items()})

loss_fn = WeightedMAELoss(weights_dict, CFG.TARGETS, CFG.DEVICE)
best_fp_indices = joblib.load(SETTINGS['SELECTED_FEATURES_PATH'])

# ===================================================================
# 3. 5折交叉验证训练
# ===================================================================
print("\n--- [步骤 3/4] 开始5折交叉验证训练 ---")
kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_wmaes = []

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    fold_output_path = os.path.join(SETTINGS['MODEL_DIR'], f"fold_{fold}")
    os.makedirs(fold_output_path, exist_ok=True)
    print(f"\n{'='*25} 开始训练第 {fold+1}/{CFG.N_FOLDS} 折 {'='*25}")
    
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    model = PolymerGNN_v12_Res(num_node_features=CFG.NUM_NODE_FEATURES, num_edge_features=CFG.NUM_EDGE_FEATURES, hidden_dim=CFG.HIDDEN_DIM, num_layers=CFG.NUM_LAYERS, tasks_fp_indices=best_fp_indices, targets=CFG.TARGETS, heads=CFG.GAT_HEADS, dropout=CFG.DROPOUT).to(CFG.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    best_val_wmae, best_epoch, patience_counter = float('inf'), 0, 0
    
    print("\n--- [阶段一] 寻找最佳轮数 ---")
    for epoch in range(1, CFG.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.DEVICE)
        val_preds_np, val_trues_np = validate_and_get_preds(model, val_loader, CFG.DEVICE)
        
        if val_preds_np.size == 0: continue
        val_preds_df = pd.DataFrame(val_preds_np, columns=CFG.TARGETS)
        val_trues_df = pd.DataFrame(val_trues_np, columns=CFG.TARGETS)
        val_wmae, _, _ = calculate_weighted_mae_official(val_trues_df, val_preds_df, CFG.TARGETS, full_train_df_for_stats)
        
        print(f"Epoch {epoch:03d}: Train wMAE Loss: {train_loss:.6f} | Val wMAE: {val_wmae:.6f}", end='\r')
        if val_wmae < best_val_wmae:
            best_val_wmae, best_epoch = val_wmae, epoch
            torch.save(model.state_dict(), os.path.join(fold_output_path, "temp_best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= CFG.ES_PATIENCE:
            print(f"\n早停触发于 epoch {epoch}。")
            break
            
    print(f"\n第 {fold+1} 折阶段一完成。最佳轮数: {best_epoch}，最佳验证wMAE: {best_val_wmae:.6f}")
    oof_wmaes.append(best_val_wmae)
    
    print("\n--- [阶段二] 学习校准器并重新训练最终模型 ---")
    model.load_state_dict(torch.load(os.path.join(fold_output_path, "temp_best_model.pth")))
    val_preds_np, val_trues_np = validate_and_get_preds(model, val_loader, CFG.DEVICE)
    calibrators = {}
    for i, task in enumerate(CFG.TARGETS):
        task_trues, task_preds = val_trues_np[:, i], val_preds_np[:, i]
        valid_mask = ~np.isnan(task_trues)
        if valid_mask.sum() > 1:
            calibrators[task] = LinearRegression().fit(task_preds[valid_mask].reshape(-1, 1), task_trues[valid_mask])
    joblib.dump(calibrators, os.path.join(fold_output_path, "calibrators.pkl"))
    print("校准模型已保存。")
    
    full_fold_dataset = torch.utils.data.Subset(full_dataset, np.concatenate([train_idx, val_idx]))
    full_fold_loader = DataLoader(full_fold_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    refit_model = PolymerGNN_v12_Res(num_node_features=CFG.NUM_NODE_FEATURES, num_edge_features=CFG.NUM_EDGE_FEATURES, hidden_dim=CFG.HIDDEN_DIM, num_layers=CFG.NUM_LAYERS, tasks_fp_indices=best_fp_indices, targets=CFG.TARGETS, heads=CFG.GAT_HEADS, dropout=CFG.DROPOUT).to(CFG.DEVICE)
    refit_optimizer = torch.optim.AdamW(refit_model.parameters(), lr=CFG.LEARNING_RATE)
    
    for epoch in tqdm(range(1, best_epoch + 1), desc=f"Refitting for {best_epoch} epochs"):
        _ = train_one_epoch(refit_model, full_fold_loader, refit_optimizer, loss_fn, CFG.DEVICE)
    torch.save(refit_model.state_dict(), os.path.join(fold_output_path, "final_refit_model.pth"))
    print(f"第 {fold+1} 折的最终模型已保存。")

# ===================================================================
# 4. 最终结果总结
# ===================================================================
print(f"\n{'='*30} 所有 {CFG.N_FOLDS} 折训练完成！ {'='*30}")
print(f"各折验证集 wMAE: {[round(l, 6) for l in oof_wmaes]}")
mean_oof_wmae, std_oof_wmae = np.mean(oof_wmaes), np.std(oof_wmaes)
print("\n" + "="*70)
print(f"? 最终 CV wMAE: {mean_oof_wmae:.6f} ± {std_oof_wmae:.6f}")
print("="*70)