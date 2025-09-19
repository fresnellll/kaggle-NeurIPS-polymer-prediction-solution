# src/predict.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import joblib
import os
import json

# PyTorch and PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# ===================================================================
# 1. 配置, 类定义与核心函数
# ===================================================================
print("--- [步骤 1/5] 加载配置，定义模型和辅助函数 ---")

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

os.makedirs(SETTINGS['SUBMISSION_DIR'], exist_ok=True)

class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES = 7, 2
    MORGAN_FP_DIM, MORGAN_FP_RADIUS = 1024, 2
    HIDDEN_DIM, NUM_LAYERS, GAT_HEADS = 384, 6, 8
    N_FOLDS = 5
    BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {CFG.DEVICE}")

def get_atom_features(atom): return [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.GetHybridization(), int(atom.GetIsAromatic()), atom.GetTotalNumHs()]

def smiles_to_periodic_graph(smiles: str):
    try: mol = Chem.MolFromSmiles(smiles, sanitize=False); Chem.SanitizeMol(mol)
    except: return None
    if mol is None: return None
    x = torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_features, star_atom_indices = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); edge_indices.extend([[i, j], [j, i]])
        bond_feats = [bond.GetBondTypeAsDouble(), int(bond.GetIsConjugated())]; edge_features.extend([bond_feats, bond_feats])
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*': star_atom_indices.append(atom.GetIdx())
    if len(star_atom_indices) == 2:
        i, j = star_atom_indices[0], star_atom_indices[1]; edge_indices.extend([[i, j], [j, i]])
        periodic_bond_feats = [-1.0, 0]; edge_features.extend([periodic_bond_feats, periodic_bond_feats])
    return Data(x=x, edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(edge_features, dtype=torch.float))

def augment_repeat_units(smiles: str, n_repeats: int = 3):
    if '*' not in smiles or n_repeats <= 1: return smiles
    try:
        monomer = Chem.MolFromSmiles(smiles)
        if not monomer: return smiles
        if len([atom.GetIdx() for atom in monomer.GetAtoms() if atom.GetSymbol() == '*']) != 2:
            core_smi = smiles.replace('[*]', ''); return f"[*]{core_smi * n_repeats}[*]"
        rxn = AllChem.ReactionFromSmarts("[*:1].[*:2]>>[*:1]-[*:2]")
        chain = monomer
        for _ in range(n_repeats - 1):
            products = rxn.RunReactants((chain, monomer))
            if not products: return smiles
            new_chain = products[0][0]; Chem.SanitizeMol(new_chain); chain = new_chain
        return Chem.MolToSmiles(chain)
    except Exception: return smiles

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

# ===================================================================
# 2. 加载并预处理测试数据 (TTA)
# ===================================================================
print("\n--- [步骤 2/5] 加载并预处理测试数据 (启用TTA) ---")
test_df = pd.read_csv(SETTINGS['TEST_CSV_PATH'])
test_smiles_all, tta_group_indices = [], []

for i, original_smi in enumerate(tqdm(test_df['SMILES'], desc="生成TTA样本")):
    tta_versions = {original_smi}
    augmented_smi = augment_repeat_units(original_smi, n_repeats=3)
    if Chem.MolFromSmiles(augmented_smi): tta_versions.add(augmented_smi)
    current_tta_group = list(tta_versions)
    test_smiles_all.extend(current_tta_group)
    tta_group_indices.extend([i] * len(current_tta_group))

print(f"TTA: {len(test_df)}个原始SMILES生成了{len(test_smiles_all)}个待预测样本。")

def process_smi_for_inference(smi):
    graph_data = smiles_to_periodic_graph(smi)
    if not graph_data: return None
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, CFG.MORGAN_FP_RADIUS, nBits=CFG.MORGAN_FP_DIM)
    graph_data.morgan_fp = torch.tensor(np.array(fp), dtype=torch.float).unsqueeze(0)
    return graph_data

test_data_list = [process_smi_for_inference(smi) for smi in tqdm(test_smiles_all, desc="处理TTA图数据")]
test_data_list_clean = [d for d in test_data_list if d is not None]

test_loader = PyGDataLoader(test_data_list_clean, batch_size=CFG.BATCH_SIZE, shuffle=False)
best_fp_indices = joblib.load(SETTINGS['SELECTED_FEATURES_PATH'])

# ===================================================================
# 3. 5折集成预测
# ===================================================================
print(f"\n--- [步骤 3/5] 开始 {CFG.N_FOLDS}-Fold 集成预测 (TTA模式) ---")
all_fold_preds = []
for fold in range(CFG.N_FOLDS):
    print(f"\n--- 正在使用 Fold-{fold} 模型进行预测 ---")
    model_path = os.path.join(SETTINGS['MODEL_DIR'], f"fold_{fold}", "final_refit_model.pth")
    calibrators_path = os.path.join(SETTINGS['MODEL_DIR'], f"fold_{fold}", "calibrators.pkl")
    
    model = PolymerGNN_v12_Res(num_node_features=CFG.NUM_NODE_FEATURES, num_edge_features=CFG.NUM_EDGE_FEATURES, hidden_dim=CFG.HIDDEN_DIM, num_layers=CFG.NUM_LAYERS, tasks_fp_indices=best_fp_indices, targets=CFG.TARGETS, heads=CFG.GAT_HEADS)
    model.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
    model.to(CFG.DEVICE)
    model.eval()
    calibrators = joblib.load(calibrators_path)
    
    fold_raw_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold-{fold} GNN预测中", leave=False):
            batch = batch.to(CFG.DEVICE)
            predictions = model(batch)
            fold_raw_preds.append(predictions.cpu().numpy())
    
    fold_raw_preds = np.concatenate(fold_raw_preds, axis=0)
    fold_calibrated_preds = np.copy(fold_raw_preds)
    for i, task in enumerate(CFG.TARGETS):
        if task in calibrators:
            task_raw_preds = fold_raw_preds[:, i].reshape(-1, 1)
            fold_calibrated_preds[:, i] = calibrators[task].predict(task_raw_preds)
    all_fold_preds.append(fold_calibrated_preds)
    print(f"Fold-{fold} 预测与校准完成！")

# ===================================================================
# 4. TTA结果聚合
# ===================================================================
print("\n--- [步骤 4/5] 平均多折预测并聚合TTA结果 ---")
final_preds_calibrated = np.mean(np.stack(all_fold_preds, axis=0), axis=0)
tta_results_df = pd.DataFrame(final_preds_calibrated, columns=CFG.TARGETS)
tta_results_df['group_index'] = tta_group_indices
final_preds_tta = tta_results_df.groupby('group_index')[CFG.TARGETS].mean().sort_index().to_numpy()
print(f"TTA平均完成，最终预测形状: {final_preds_tta.shape}")

# ===================================================================
# 5. 生成提交文件
# ===================================================================
print("\n--- [步骤 5/5] 生成提交文件 ---")
submission_df = pd.read_csv(os.path.join(SETTINGS['RAW_DATA_DIR'], "sample_submission.csv"))

if final_preds_tta.shape[0] == len(submission_df):
    submission_df[CFG.TARGETS] = final_preds_tta
else:
    print(f"[警告] 预测结果 ({final_preds_tta.shape[0]}) 与提交文件行数 ({len(submission_df)}) 不匹配！")

submission_df.to_csv(SETTINGS['SUBMISSION_FILE'], index=False)
print(f"submission.csv 文件已成功生成于: {SETTINGS['SUBMISSION_FILE']}")
print("提交文件预览:")
print(submission_df.head())