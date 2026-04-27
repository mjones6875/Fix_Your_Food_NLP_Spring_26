# ================================================================================
# Advice Improvement – DistilBERT Multi-label Classifier
# ================================================================================
# Fine-tunes DistilBERT to classify each McDonald's review across three axes:
#   - OrderAccuracy   (e.g. Missing item, Incorrect)
#   - CustomizationError (e.g. Wrong sauce, Missing ingredients)
#   - FoodQuality     (e.g. Stale, Cold, Burnt)
#
# Trains on TestSampleData.csv, runs inference on mcdonaldsfoodreviews.csv.
# Keyword overrides applied after model predictions for high-confidence cases.
# ================================================================================

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder

# GPU if available, otherwise CPU
processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths — used as defaults in run_advice_improvement()
TRAIN_CSV = "TestSampleData.csv"          # was "foodreviewdata.csv" in original Colab version
INFERENCE_CSV = "mcdonaldsfoodreviews.csv"  # was "mcdonaldsfoodreview.csv" (missing 's') in original
OUTPUT_CSV = "mcdonaldsfoodreviews_advice.csv"  # never overwrites source data

# ================================================================================
# Keyword Safety Net
# ================================================================================
# These override model predictions when a keyword is an unambiguous signal.
# Prevents soft model confidence from misclassifying clear-cut cases.

QUALITY_KEYWORDS = {
    "Stale": ["never fresh", "not fresh", "stale", "hard", "old"],
    "Cold": ["not hot", "cold", "ice cold", "freezing"],
    "Hot": ["hot", "steaming", "piping", "freshly made"],
    "Burnt": ["burnt", "burned", "charred"],
    "Undercooked": ["pink", "raw", "undercooked"],
    "Soggy": ["soggy", "mushy", "wet"]
}

CUSTOM_KEYWORDS = {
    "None": ["no special orders", "standard order"],
    "Wrong sauce": ["sauce", "mayo", "ketchup", "mustard", "ranch"],
    "Missing ingredients": ["no cheese", "no onion", "no pickle", "without"],
    "Wrong customizations": ["asked for", "requested", "instead of", "plain"]
}

ACCURACY_KEYWORDS = {
    "Missing item": ["missing", "forgot", "didnt get", "not in the bag"],
    "Incorrect": ["wrong item", "wrong sandwich", "not what i ordered"]
}

def apply_overrides(text, current_label, rules):
    """Return the first keyword-matched label, or current_label if no match."""
    t = str(text).lower()
    for label, keywords in rules.items():
        if any(k in t for k in keywords):
            return label
    return current_label


# ================================================================================
# Dataset & Model
# ================================================================================

class ReviewDataset(Dataset):
    # tokenizer passed in so it isn't a module-level side effect on import
    def __init__(self, comments, tokenizer, orders=None, customs=None, qualities=None):
        self.comments = comments
        self.tokenizer = tokenizer
        self.orders = orders
        self.customs = customs
        self.qualities = qualities

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, i):
        tok = self.tokenizer(
            str(self.comments[i]),
            truncation=True, padding="max_length", max_length=64, return_tensors="pt"
        )
        item = {"ids": tok["input_ids"].squeeze(0), "mask": tok["attention_mask"].squeeze(0)}
        if self.orders is not None:
            item.update({
                "o": torch.tensor(self.orders[i]),
                "c": torch.tensor(self.customs[i]),
                "q": torch.tensor(self.qualities[i])
            })
        return item

class FeedbackAI(nn.Module):
    def __init__(self, n_order, n_custom, n_quality):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        h = self.bert.config.hidden_size
        # separate classification head for each label axis
        self.o_head = nn.Linear(h, n_order)
        self.c_head = nn.Linear(h, n_custom)
        self.q_head = nn.Linear(h, n_quality)

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.o_head(x), self.c_head(x), self.q_head(x)


# ================================================================================
# run_advice_improvement() — entry point for main.py
# All script logic lives here so this module is safely importable without auto-running.
# ================================================================================

def run_advice_improvement(
    train_csv=TRAIN_CSV,
    inference_csv=INFERENCE_CSV,
    output_csv=OUTPUT_CSV
):
    # === Load training data ===
    train_df = pd.read_csv(train_csv, encoding="utf-8-sig")  # utf-8-sig preserves emojis/special chars

    # Fill missing label columns with safe defaults before encoding
    for col, val in [("OrderAccuracy", "Correct"),
                     ("CustomizationError", "None"),
                     ("FoodQuality", "N/A")]:
        train_df[col] = train_df[col].fillna(val)

    # Fit label encoders on training data
    order_enc = LabelEncoder().fit(train_df["OrderAccuracy"])
    custom_enc = LabelEncoder().fit(train_df["CustomizationError"])
    quality_enc = LabelEncoder().fit(train_df["FoodQuality"])

    # === Build tokenizer & model ===
    # Initialized here (not at import time) to avoid downloading DistilBERT on import
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = FeedbackAI(
        n_order=len(order_enc.classes_),
        n_custom=len(custom_enc.classes_),
        n_quality=len(quality_enc.classes_)
    ).to(processing_device)

    # === Train ===
    loader = DataLoader(
        ReviewDataset(
            train_df["Comment"].values,
            tokenizer,
            order_enc.transform(train_df["OrderAccuracy"]),
            custom_enc.transform(train_df["CustomizationError"]),
            quality_enc.transform(train_df["FoodQuality"])
        ),
        batch_size=32, shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        for i, b in enumerate(loader):
            optimizer.zero_grad()
            po, pc, pq = model(b["ids"].to(processing_device), b["mask"].to(processing_device))
            loss = (loss_fn(po, b["o"].to(processing_device))
                    + loss_fn(pc, b["c"].to(processing_device))
                    + loss_fn(pq, b["q"].to(processing_device)))
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {i} processed")

    # === Inference on production data ===
    mcd_df = pd.read_csv(inference_csv, encoding="utf-8-sig")
    raw_comments = mcd_df["Comment"].fillna("").astype(str).values

    inf_loader = DataLoader(ReviewDataset(raw_comments, tokenizer), batch_size=32)

    out_o, out_c, out_q = [], [], []
    model.eval()
    with torch.no_grad():
        for b in inf_loader:
            po, pc, pq = model(b["ids"].to(processing_device), b["mask"].to(processing_device))
            out_o.extend(order_enc.inverse_transform(po.argmax(1).cpu().numpy()))
            out_c.extend(custom_enc.inverse_transform(pc.argmax(1).cpu().numpy()))
            out_q.extend(quality_enc.inverse_transform(pq.argmax(1).cpu().numpy()))

    # === Apply keyword overrides and write output ===
    final_o, final_c, final_q = [], [], []
    for i, txt in enumerate(raw_comments):
        final_o.append(apply_overrides(txt, out_o[i], ACCURACY_KEYWORDS))
        final_c.append(apply_overrides(txt, out_c[i], CUSTOM_KEYWORDS))
        final_q.append(apply_overrides(txt, out_q[i], QUALITY_KEYWORDS))

    mcd_df["OrderAccuracy"] = final_o
    mcd_df["CustomizationError"] = final_c
    mcd_df["FoodQuality"] = final_q

    mcd_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Export] Written to '{output_csv}' ({len(mcd_df)} rows)")

    return output_csv


# === allows direct run or clean import by main.py ===
if __name__ == "__main__":
    run_advice_improvement()
