# Kaggle NeurIPS 2025 - Open Polymer Prediction: 3rd Solution

This repository contains the complete code and documentation for the winning solution to the Kaggle Open Polymer Prediction competition. The approach is centered around a Graph Attention Network (GATv2) enhanced with task-specific feature engineering and data augmentation.

## 1. Hardware & Software

*   **Hardware (Recommended):**
    *   **CPU:** Intel(R) Core(TM) Utral 9 185H
    *   **Memory:** 32GB RAM
    *   **GPU:** NVIDIA RTX 4070 ti super 
*   **Software:**
    *   **OS:** Linux (Ubuntu 22.04 LTS via WSL2 tested)
    *   **Environment:** Python 3.11 managed with Miniconda
    *   **Key Libraries:** PyTorch 2.6, PyTorch Geometric 2.6, RDKit 2025.3.3

## 2. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [your-repo-url]
    cd kaggle_nips_2025_polymer_prediction
    ```

2.  **Create Conda Environment:**
    It is highly recommended to use Conda to manage dependencies.
    ```bash
    # Create a new conda environment named 'polymer_gnn'
    conda create -n polymer_gnn python=3.11 -y
    conda activate polymer_gnn
    ```

3.  **Install Dependencies (IMPORTANT: Multi-step process):**
    Due to build dependencies, the installation must be done in a specific order.

    **Step 3.1: Install Core Libraries from requirements.txt**
    First, install PyTorch, RDKit, and other core packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

    **Step 3.2: Install PyTorch Geometric Libraries**
    Next, install PyG and its dependencies using the official pre-compiled wheels, which avoids build errors. This command installs versions compatible with PyTorch 2.6.0 and CPU. Replace `+cpu` with your specific CUDA version (e.g., `+cu121`) if you have a GPU.
    ```bash
    # For CPU-only installation (as tested)
    pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
    pip install torch_geometric
    ```
    *Note: The command `pip install torch_spline_conv` from your testing is often included for completeness, but since our model (GATv2) does not use it, installing `torch_geometric` is sufficient and will pull in necessary dependencies.*

4.  **Download Competition Data:**
    Download the official competition data from the [Kaggle competition page](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data). You can use our data in /data/raw/ and processed data in /data/processed/all_augmented_master.pt

## 3. How to Run the Full Pipeline

The entire solution is executed via three sequential scripts. The commands are also listed in `entry_points.md`. 

**IMPORTANT:** Ensure you are in the project's root directory (`Kaggle_Polymer_GNN_Solution/`) when running all the following commands, using the `python src/script_name.py` format.

### Step 1: Prepare Data
This script reads the raw `.csv` files, performs chemical data augmentation, selects features, and saves the final processed dataset.
```bash
python src/prepare_data.py
```

### Step 2: Train Model
This script trains the 5-fold cross-validation models using the processed data.
```bash
python src/train.py
```

### Step 3: Generate Predictions
This script loads the 5 trained models and generates the final submission file.
```bash
python src/predict.py
```
*   **Output:** `submission.csv` will be created in the `./output/` directory.

## 4. Key Assumptions

*   The installation instructions assume a CPU-based environment but can be adapted for GPU by changing the PyG installation command.
*   The `SETTINGS.json` file is the single source of truth for all I/O paths.
*   All Python scripts must be executed from the project's root directory.