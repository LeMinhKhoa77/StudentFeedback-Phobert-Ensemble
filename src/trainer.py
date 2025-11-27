import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dataset import MultiTaskDataset, collate_fn
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

class PhoBERTMultiTask(nn.Module):
    def __init__(self, model_name, num_sentiment, num_topic):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment)
        self.topic_head = nn.Linear(hidden_size, num_topic)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.sentiment_head(pooled), self.topic_head(pooled)

def make_loader(df, tokenizer, batch_size=16, max_len=256, shuffle=True):
    ds = MultiTaskDataset(df, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def train_one_fold(train_df, val_df, tokenizer, model_name, num_sentiment, num_topic, fold_dir,
                   device, epochs=3, batch_size=16, lr=2e-5):
    os.makedirs(fold_dir, exist_ok=True)
    train_loader = make_loader(train_df, tokenizer, batch_size=batch_size)
    val_loader = make_loader(val_df, tokenizer, batch_size=batch_size, shuffle=False)

    model = PhoBERTMultiTask(model_name, num_sentiment, num_topic).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps),
                                                num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentiment_labels = batch["sentiment"].to(device)
            topic_labels = batch["topic"].to(device)

            s_logits, t_logits = model(input_ids, attention_mask)
            loss = loss_fn(s_logits, sentiment_labels) + loss_fn(t_logits, topic_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        s_preds, s_trues, t_preds, t_trues = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                sentiment_labels = batch["sentiment"].to(device)
                topic_labels = batch["topic"].to(device)
                s_logits, t_logits = model(input_ids, attention_mask)
                s_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
                s_trues.extend(sentiment_labels.cpu().numpy())
                t_preds.extend(torch.argmax(t_logits, dim=1).cpu().numpy())
                t_trues.extend(topic_labels.cpu().numpy())

        s_f1 = f1_score(s_trues, s_preds, average="macro")
        t_f1 = f1_score(t_trues, t_preds, average="macro")
        print(f"Epoch {epoch} -> Loss: {avg_loss:.4f} | Sentiment F1: {s_f1:.4f} | Topic F1: {t_f1:.4f}")

        # Save best model
        if s_f1 + t_f1 > best_f1:
            best_f1 = s_f1 + t_f1
            torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
            tokenizer.save_pretrained(fold_dir)
            print(f"[SAVE] Best model saved to {fold_dir}")

    return best_f1, fold_dir

def ensemble_predict(fold_dirs, df, tokenizer, device, model_name, num_sentiment, num_topic, batch_size=32):
    loader = make_loader(df, tokenizer, batch_size=batch_size, shuffle=False)
    s_logits_accum, t_logits_accum = None, None

    for fd in fold_dirs:
        if not os.path.exists(fd):
            continue
        model = PhoBERTMultiTask(model_name, num_sentiment, num_topic).to(device)
        model.load_state_dict(torch.load(os.path.join(fd, "model.pt")))
        model.eval()

        s_fold_logits, t_fold_logits = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
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
