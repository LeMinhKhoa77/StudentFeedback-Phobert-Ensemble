import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = str(self.df.loc[idx, "text"])
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "sentiment": torch.tensor(int(self.df.loc[idx, "sentiment"])),
            "topic": torch.tensor(int(self.df.loc[idx, "topic"]))
        }
        return item

def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    sentiment = torch.tensor([x["sentiment"] for x in batch])
    topic = torch.tensor([x["topic"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "sentiment": sentiment, "topic": topic}
