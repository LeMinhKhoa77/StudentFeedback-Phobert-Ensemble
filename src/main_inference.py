# main_inference.py
import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from trainer import PhoBERTMultiTask  # từ trainer.py
from preprocess import clean_text
from torch.utils.data import Dataset, DataLoader

# --- Cấu hình ---
MODEL_NAME = "vinai/phobert-base-v2"
DEVICE = torch.device("cpu")  # luôn CPU
BATCH_SIZE = 32

FOLD_DIRS = [
    "saved_models/fold_1",
    "saved_models/fold_2",
    "saved_models/fold_3",
    "saved_models/fold_4",
    "saved_models/fold_5",
]

with open("saved_models/sentiment_map.json", "r", encoding="utf-8") as f:
    sentiment_map = json.load(f)
with open("saved_models/topic_map.json", "r", encoding="utf-8") as f:
    topic_map = json.load(f)

NUM_SENTIMENT = len(sentiment_map)  # 2
NUM_TOPIC = len(topic_map)          # 4

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# --- Dataset & DataLoader ---
class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, "clean_text"])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

def make_loader(df, tokenizer, batch_size=BATCH_SIZE):
    ds = InferenceDataset(df, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

# --- Ensemble Predict CPU ---
def ensemble_predict_cpu(fold_dirs, df, tokenizer, model_name, num_sentiment, num_topic, batch_size=BATCH_SIZE):
    loader = make_loader(df, tokenizer, batch_size)
    s_logits_accum, t_logits_accum = None, None

    for fd in fold_dirs:
        model_path = os.path.join(fd, "model.pt")
        if not os.path.exists(model_path):
            print(f"[WARN] Không tìm thấy {model_path}, bỏ qua.")
            continue

        model = PhoBERTMultiTask(model_name, num_sentiment, num_topic).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        s_fold_logits, t_fold_logits = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                s_logits, t_logits = model(input_ids, attention_mask)
                s_fold_logits.append(s_logits.cpu())
                t_fold_logits.append(t_logits.cpu())

        s_fold_logits = torch.cat(s_fold_logits).numpy()
        t_fold_logits = torch.cat(t_fold_logits).numpy()
        s_logits_accum = s_fold_logits if s_logits_accum is None else s_logits_accum + s_fold_logits
        t_logits_accum = t_fold_logits if t_logits_accum is None else t_logits_accum + t_fold_logits
        del model
        torch.cuda.empty_cache()

    s_avg = s_logits_accum / len(fold_dirs)
    t_avg = t_logits_accum / len(fold_dirs)
    s_pred = s_avg.argmax(axis=1)
    t_pred = t_avg.argmax(axis=1)
    return s_pred, t_pred

# --- Input / Output ---
INPUT_FILE = "student_feedback.csv"
OUTPUT_FILE = "result.csv"

df = pd.read_csv(INPUT_FILE)
df["clean_text"] = df["text"].apply(clean_text)

print("⚡ Bắt đầu dự đoán với CPU...")

s_pred, t_pred = ensemble_predict_cpu(
    FOLD_DIRS,
    df,
    tokenizer,
    MODEL_NAME,
    NUM_SENTIMENT,
    NUM_TOPIC
)

df["sentiment_pred"] = s_pred
df["topic_pred"] = t_pred
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"✔ Kết quả đã được lưu tại {OUTPUT_FILE}")
