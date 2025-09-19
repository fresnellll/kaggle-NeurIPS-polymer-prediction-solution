#### **2. `entry_points.md` **
```markdown
# Entry Points for Solution Pipeline

All commands should be run from the project's root directory.

## 1. Data Preparation
# Pre-processes raw data, performs augmentation and feature selection.
python src/prepare_data.py

## 2. Model Training
# Trains the 5-fold cross-validation models.
python src/train.py

## 3. Prediction Generation
# Generates the final submission.csv using the trained models.
python src/predict.py
```

