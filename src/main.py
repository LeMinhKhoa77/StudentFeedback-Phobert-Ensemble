from preprocess import load_and_preprocess
from trainer import train_one_fold, ensemble_predict
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
import os

DATA_FILES = ["data/dataframe1.csv", "data/dataframe2.csv"]
MODEL_NAME = "vinai/phobert-base-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data và tạo mapping
df, sentiment_map, topic_map = load_and_preprocess(DATA_FILES)

NUM_SENTIMENT = len(sentiment_map)
NUM_TOPIC = len(topic_map)

# Train/Dev/Test split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["sentiment"])
dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["sentiment"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
SAVE_ROOT = "saved_models"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 5-fold CV
NUM_FOLDS = 5
kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_dirs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df["sentiment"]), 1):
    print(f"\n=== Fold {fold} ===")
    fold_train = train_df.iloc[train_idx].reset_index(drop=True)
    fold_val = train_df.iloc[val_idx].reset_index(drop=True)

    fold_dir = os.path.join(SAVE_ROOT, f"fold_{fold}")

    best_f1, saved_dir = train_one_fold(
        fold_train,
        fold_val,
        tokenizer,
        MODEL_NAME,
        NUM_SENTIMENT,
        NUM_TOPIC,
        fold_dir,
        DEVICE,
        epochs=3,
        batch_size=16
    )

    fold_dirs.append(saved_dir)

# ==========================
# Ensemble on Dev set
# ==========================

print("\nEvaluating ensemble on Dev set")

s_dev_pred, t_dev_pred = ensemble_predict(
    fold_dirs,
    dev_df,
    tokenizer,
    DEVICE,
    NUM_SENTIMENT,   # thêm tham số bắt buộc
    NUM_TOPIC        # thêm tham số bắt buộc
)

print("Dev sentiment F1:", f1_score(dev_df["sentiment"], s_dev_pred, average="macro"))
print("Dev topic F1:", f1_score(dev_df["topic"], t_dev_pred, average="macro"))

# ==========================
# Ensemble on Test set
# ==========================

print("\nEvaluating ensemble on Test set")

s_test_pred, t_test_pred = ensemble_predict(
    fold_dirs,
    test_df,
    tokenizer,
    DEVICE,
    NUM_SENTIMENT,
    NUM_TOPIC
)

print("Test sentiment F1:", f1_score(test_df["sentiment"], s_test_pred, average="macro"))
print("Test topic F1:", f1_score(test_df["topic"], t_test_pred, average="macro"))
