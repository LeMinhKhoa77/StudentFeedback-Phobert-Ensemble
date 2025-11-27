import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import gradio as gr

# =========================
# CẤU HÌNH
# =========================
MODEL_NAME = "vinai/phobert-base-v2"
FOLD_DIRS = [f"/content/drive/MyDrive/saved_models/fold_{i}" for i in range(1, 6)]
NUM_SENTIMENT = 2
NUM_TOPIC = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MÔ HÌNH
# =========================
class PhoBERTMultiTask(nn.Module):
    def __init__(self, model_name, num_sentiment, num_topic):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.sentiment_head = nn.Linear(768, num_sentiment)
        self.topic_head = nn.Linear(768, num_topic)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1]
        return self.sentiment_head(pooled), self.topic_head(pooled)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# =========================
# DATASET
# =========================
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=128, return_tensors='pt'
        )
        return {
            "input_ids": enc['input_ids'].squeeze(0),
            "attention_mask": enc['attention_mask'].squeeze(0)
        }

# =========================
# LABEL MAPPING
# =========================
TOPIC_MAP = {0: "Cơ sở vật chất", 1: "Giảng viên", 2: "Khác", 3: "Chương trình đào tạo"}
SENTIMENT_MAP = {0: "Negative", 1: "Positive"}

# =========================
# ENSEMBLE PREDICT (5-fold)
# =========================
def ensemble_predict(texts):
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=8)

    fold_s_preds, fold_t_preds = [], []

    for fold_dir in FOLD_DIRS:
        model = PhoBERTMultiTask(MODEL_NAME, NUM_SENTIMENT, NUM_TOPIC).to(DEVICE)
        ckpt_path = os.path.join(fold_dir, "model.pt")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)
        model.eval()

        s_preds, t_preds = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)

                s_logits, t_logits = model(ids, mask)

                s_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
                t_preds.extend(torch.argmax(t_logits, dim=1).cpu().numpy())

        fold_s_preds.append(s_preds)
        fold_t_preds.append(t_preds)

    # Voting
    s_preds = np.array(fold_s_preds).T
    t_preds = np.array(fold_t_preds).T

    final_s = [np.bincount(row).argmax() for row in s_preds]
    final_t = [np.bincount(row).argmax() for row in t_preds]

    return final_s, final_t

# =========================
# PHÂN TÍCH FILE CSV
# =========================
def analyze_file(csv_file):
    df = pd.read_csv(csv_file.name)

    if "text" not in df.columns:
        return pd.DataFrame({"Error": ["CSV phải có cột 'text'"]}), None, None

    s_preds, t_preds = ensemble_predict(df["text"].tolist())

    df["sentiment_pred"] = s_preds
    df["topic_pred"] = t_preds
    df["topic_name"] = df["topic_pred"].map(TOPIC_MAP)
    df["sentiment_name"] = df["sentiment_pred"].map(SENTIMENT_MAP)

    # Thống kê
    summary = []
    for topic, group in df.groupby("topic_name"):
        total = len(group)
        pos = (group["sentiment_pred"] == 1).sum()
        neg = (group["sentiment_pred"] == 0).sum()

        summary.append({
            "Topic": topic,
            "Total": total,
            "Positive": pos,
            "Positive (%)": round(pos / total * 100, 1),
            "Negative": neg,
            "Negative (%)": round(neg / total * 100, 1)
        })

    stats_df = pd.DataFrame(summary)

    # Biểu đồ
    fig, ax = plt.subplots(figsize=(8, 5))
    topics = stats_df["Topic"]
    pos_pct = stats_df["Positive (%)"]
    neg_pct = stats_df["Negative (%)"]

    ax.bar(topics, pos_pct, label="Positive (%)", color="skyblue")
    ax.bar(topics, neg_pct, bottom=pos_pct, label="Negative (%)", color="salmon")

    ax.set_ylabel("Percentage")
    ax.set_title("Sentiment Distribution by Topic")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    return df, stats_df, fig

# =========================
# GRADIO APP
# =========================
iface = gr.Interface(
    fn=analyze_file,
    inputs=gr.File(file_types=[".csv"]),
    outputs=[
        gr.Dataframe(label="Predictions"),
        gr.Dataframe(label="Statistics"),
        gr.Plot(label="Sentiment Distribution Chart")
    ],
    title="Student Feedback Analyzer (5-Fold Ensemble)",
    description="Upload a CSV with 'text' column to analyze sentiment and topic."
)

iface.launch(share=True)
