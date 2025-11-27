# main_test.py
import os
import torch
from transformers import AutoTokenizer
import pandas as pd
from trainer import train_one_fold
from preprocess import load_and_preprocess

# --------------------------
# Cấu hình test nhanh
# --------------------------
MODEL_NAME = "vinai/phobert-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SENTIMENT = 2  # ví dụ: 0=negative, 1=positive
NUM_TOPIC = 4      # 0,1,2,3
BATCH_SIZE = 8
EPOCHS = 1
FOLD_DIR = "fold_test"

# --------------------------
# Load và tiền xử lý dữ liệu
# --------------------------
DATA_FILES = ["data/dataframe1.csv", "data/dataframe2.csv"]

print("Loading and preprocessing data...")
df = pd.concat([pd.read_csv(f) for f in DATA_FILES], ignore_index=True)
df, sentiment_map, topic_map = load_and_preprocess(DATA_FILES)

# Lấy 500 dòng để test nhanh
df_small = df.sample(n=500, random_state=42).reset_index(drop=True)

# Chia train/val tạm thời 80/20
train_size = int(len(df_small) * 0.8)
fold_train = df_small[:train_size]
fold_val = df_small[train_size:]

# --------------------------
# Khởi tạo tokenizer
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --------------------------
# Train 1 fold test
# --------------------------
print("Running test training for 1 fold, 1 epoch...")
best_f1, saved_dir = train_one_fold(
    train_df=fold_train,
    val_df=fold_val,
    tokenizer=tokenizer,
    model_name=MODEL_NAME,
    num_sentiment=NUM_SENTIMENT,
    num_topic=NUM_TOPIC,
    fold_dir=FOLD_DIR,
    device=DEVICE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print(f"Test finished. Best combined F1: {best_f1:.4f}")
print(f"Model saved to: {saved_dir}")
