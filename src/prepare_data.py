# src/prepare_data.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from functools import reduce
import json
import joblib

# PyTorch and PyG
import torch
from torch_geometric.data import Data, InMemoryDataset

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# Scikit-learn for feature selection
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings("ignore")
tqdm.pandas()

# ===================================================================
# 1. 配置与核心函数
# ===================================================================
print("--- [步骤 1/5] 加载配置和定义核心函数 ---")

with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

# 创建所有需要的输出目录
for dir_path in [SETTINGS['PROCESSED_DATA_DIR'], SETTINGS['ASSETS_DIR']]:
    os.makedirs(dir_path, exist_ok=True)

SMILES_COL = 'SMILES'
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
MORGAN_FP_DIM = 1024
MORGAN_FP_RADIUS = 2
NUM_BEST_FEATURES = 50

# --- 特征提取、图构建与数据增强函数 (来自您的原始代码) ---
def get_atom_features(atom):
    return [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.GetHybridization(), int(atom.GetIsAromatic()), atom.GetTotalNumHs()]

def get_bond_features(bond):
    return [bond.GetBondTypeAsDouble(), int(bond.GetIsConjugated())]

def smiles_to_periodic_graph(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol)
    except: return None
    if mol is None: return None
    x = torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_features, star_atom_indices = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        bond_feats = get_bond_features(bond)
        edge_features.extend([bond_feats, bond_feats])
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*': star_atom_indices.append(atom.GetIdx())
    if len(star_atom_indices) == 2:
        i, j = star_atom_indices[0], star_atom_indices[1]
        edge_indices.extend([[i, j], [j, i]])
        periodic_bond_feats = [-1.0, 0]
        edge_features.extend([periodic_bond_feats, periodic_bond_feats])
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
            if not products: break
            product_mol = products[0][0]
            Chem.SanitizeMol(product_mol)
            chain = product_mol
        return Chem.MolToSmiles(chain)
    except Exception: return smiles

# ===================================================================
# 2. 数据加载与合并
# ===================================================================
print("\n--- [步骤 2/5] 加载并合并原始训练数据 ---")
dfs = []
raw_data_dir = SETTINGS['RAW_DATA_DIR']
data_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv') and f not in ['test.csv', 'sample_submission.csv']]

for filename in data_files:
    try:
        df = pd.read_csv(os.path.join(raw_data_dir, filename))
        target_name = os.path.splitext(filename)[0]
        if target_name in TARGETS:
            df.rename(columns={df.columns[0]: SMILES_COL, df.columns[1]: target_name}, inplace=True)
            dfs.append(df[[SMILES_COL, target_name]])
    except Exception as e:
        print(f"警告: 无法处理文件 {filename}. 错误: {e}")

main_df = reduce(lambda left, right: pd.merge(left, right, on=SMILES_COL, how='outer'), dfs)
main_df.dropna(subset=[SMILES_COL], inplace=True)
main_df = main_df.groupby(SMILES_COL).first().reset_index()
print(f"数据合并完成，得到 {len(main_df)} 个独特的SMILES。")

# ===================================================================
# 3. 特征选择 (基于原始未增强数据)
# ===================================================================
print(f"\n--- [步骤 3/5] 为每个任务选择前 {NUM_BEST_FEATURES} 个摩根指纹 ---")
fps = main_df[SMILES_COL].progress_apply(lambda s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), MORGAN_FP_RADIUS, nBits=MORGAN_FP_DIM) if Chem.MolFromSmiles(s) else None)
fp_df = pd.DataFrame([list(fp) if fp else [0]*MORGAN_FP_DIM for fp in fps], index=main_df.index)

best_fp_indices = {}
for task in TARGETS:
    task_df = main_df[[task]].join(fp_df).dropna()
    X = task_df.drop(columns=[task])
    y = task_df[task]
    if len(y) < 2: continue
    
    k = min(NUM_BEST_FEATURES, X.shape[1])
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    best_fp_indices[task] = selector.get_support(indices=True).tolist()
    print(f"任务 '{task}' 的前 {k} 个最佳指纹索引已选出。")

joblib.dump(best_fp_indices, SETTINGS['SELECTED_FEATURES_PATH'])
print(f"所有任务的最佳指纹索引已保存到: {SETTINGS['SELECTED_FEATURES_PATH']}")

# ===================================================================
# 4. 数据增强与图转换 (修正后的核心逻辑)
# ===================================================================
print("\n--- [步骤 4/5] 对全量数据进行增强并转换为PyG图对象 ---")
data_list = []
for _, row in tqdm(main_df.iterrows(), total=len(main_df), desc="处理SMILES并增强"):
    original_smiles = row[SMILES_COL]
    targets = [row.get(t, np.nan) for t in TARGETS]
    y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)

    def process_single_smiles(smi, label):
        graph_data = smiles_to_periodic_graph(smi)
        if not graph_data: return None
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_FP_RADIUS, nBits=MORGAN_FP_DIM)
            graph_data.morgan_fp = torch.tensor(np.array(fp), dtype=torch.float).unsqueeze(0)
        else:
            graph_data.morgan_fp = torch.zeros(1, MORGAN_FP_DIM, dtype=torch.float)
        graph_data.y = label
        return graph_data

    # 处理原始SMILES
    processed_original = process_single_smiles(original_smiles, y)
    if processed_original:
        data_list.append(processed_original)
    
    # 处理增强后的SMILES
    augmented_smiles = augment_repeat_units(original_smiles, n_repeats=3)
    if augmented_smiles != original_smiles:
        processed_augmented = process_single_smiles(augmented_smiles, y)
        if processed_augmented:
            data_list.append(processed_augmented)

print(f"数据处理完成，共生成 {len(data_list)} 个图对象 (包含原始和增强数据)。")

# ===================================================================
# 5. 保存处理后的数据集
# ===================================================================
print("\n--- [步骤 5/5] 保存最终的PyG数据集 ---")
class TempDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data, self.slices = self.collate(data_list)
        
temp_dataset = TempDataset(data_list)
torch.save((temp_dataset.data, temp_dataset.slices), SETTINGS['PROCESSED_TRAIN_DATA_PATH'])
print(f"全量增强数据集已保存到: {SETTINGS['PROCESSED_TRAIN_DATA_PATH']}")
print("\n数据准备流程完成！")