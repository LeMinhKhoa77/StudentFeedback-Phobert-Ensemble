import pandas as pd
import re
import unicodedata
from tqdm import tqdm

EMOTICON_MAP = {
    r':\)+': ' positive_emoticon ',
    r':\(+': ' negative_emoticon ',
    r'>:\(+': ' angry_emoticon ',
    r':d': ' bigsmile_emoticon ',
    r'<3': ' love_emoticon ',
    r':v': ' pacman_emoticon ',
    r'xd+': ' laugh_emoticon ',
}

ABBREV_MAP = {
    r'\bko\b': ' không ',
    r'\bdc\b': ' được ',
    r'\bvs\b': ' với ',
}
    


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text.lower())
    for pat, rep in EMOTICON_MAP.items():
        text = re.sub(pat, rep, text)
    for pat, rep in ABBREV_MAP.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(file_paths):
    dfs = [pd.read_csv(p) for p in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    tqdm.pandas(desc="Cleaning text")
    df["text"] = df["text"].progress_apply(clean_text)
    df = df[df["text"].str.split().str.len() >= 2].reset_index(drop=True)
    
    # Label encoding
    sentiment_map = {lab:i for i, lab in enumerate(sorted(df["sentiment"].unique()))}
    topic_map = {lab:i for i, lab in enumerate(sorted(df["topic"].unique()))}
    df["sentiment"] = df["sentiment"].map(sentiment_map)
    df["topic"] = df["topic"].map(topic_map)
    return df, sentiment_map, topic_map
