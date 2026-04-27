# ================================================================================
# FixYourFood – Topic Categorization (TF-IDF + Logistic Regression)
# ================================================================================
# Assigns each McDonald's review one topic label using supervised ML.
# Trains on TestSampleData.csv, runs inference on mcdonaldsfoodreviews.csv.
# ================================================================================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -------- STANDARDIZED TOPIC LABELS --------
LABELS = [
    "FOOD_QUALITY",
    "STAFF_SERVICE",
    "TIMELINESS_SPEED",
    "CLEANLINESS",
    "ENVIRONMENT_ATMOSPHERE",
    "CUSTOMIZATION_ERROR"
]

# -------- CATEGORY MAPPING --------
# Maps raw CSV categories to standardized labels.
# Both "Order accuracy" and "Customization error" map to CUSTOMIZATION_ERROR
# because they represent the same class of fulfillment issue.
CATEGORY_MAPPING = {
    "Food quality": "FOOD_QUALITY",
    "food quality": "FOOD_QUALITY",
    "Staff service": "STAFF_SERVICE",
    "staff service": "STAFF_SERVICE",
    "Timeliness": "TIMELINESS_SPEED",
    "timeliness": "TIMELINESS_SPEED",
    "Cleanliness": "CLEANLINESS",
    "cleanliness": "CLEANLINESS",
    "Environment": "ENVIRONMENT_ATMOSPHERE",
    "environment": "ENVIRONMENT_ATMOSPHERE",
    "Customization error": "CUSTOMIZATION_ERROR",
    "customization error": "CUSTOMIZATION_ERROR",
    "Order accuracy": "CUSTOMIZATION_ERROR",
    "order accuracy": "CUSTOMIZATION_ERROR",
}

MIN_WORDS = 5   # reviews under this threshold are excluded from training and inference
RANDOM_SEED = 42

# File paths — used as defaults in run_topic_categorization()
TRAIN_CSV = "TestSampleData.csv"
INFERENCE_CSV = "mcdonaldsfoodreviews.csv"
OUTPUT_CSV = "mcdonaldsfoodreviews_predicted.csv"  # never overwrites source data


# ================================================================================
# Helper Functions
# ================================================================================

def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len([w for w in text.strip().split() if w])

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())

def normalize_category(category_str: str) -> str:
    """Returns standardized label or None if unmapped (unmapped rows are dropped)."""
    if not isinstance(category_str, str):
        return None
    return CATEGORY_MAPPING.get(category_str.strip(), None)

def load_and_filter_reviews(csv_path: str, text_col: str = "Comment",
                             category_col: str = "Category") -> pd.DataFrame:
    """Load labeled CSV, clean text, filter short/unmapped/duplicate reviews."""
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}'. Found: {list(df.columns)}")
    if category_col not in df.columns:
        raise ValueError(f"Expected column '{category_col}'. Found: {list(df.columns)}")

    df[text_col] = df[text_col].astype(str).apply(basic_clean)
    df["word_count"] = df[text_col].apply(word_count)
    df["topic_label"] = df[category_col].apply(normalize_category)

    initial_count = len(df)
    df = df[df["topic_label"].notna()].copy()
    df = df[df["word_count"] >= MIN_WORDS].copy()
    df = df.drop_duplicates(subset=[text_col], keep="first").copy()
    df.reset_index(drop=True, inplace=True)

    print(f"[Data Filtering] {initial_count} rows -> {len(df)} kept "
          f"({initial_count - len(df)} removed)")
    return df


# ================================================================================
# Model Training & Evaluation
# ================================================================================

def train_and_evaluate_topic_model(text_list, label_list, seed: int = RANDOM_SEED):
    """
    Train TF-IDF + Logistic Regression on an 80/20 stratified split.
    Returns (pipeline, confusion_matrix_df, (X_test, y_test, y_pred)).
    """
    assert len(text_list) == len(label_list) and len(text_list) > 0
    invalid = set(label_list) - set(LABELS)
    if invalid:
        raise ValueError(f"Invalid labels: {invalid}")

    X_train, X_test, y_train, y_test = train_test_split(
        text_list, label_list, test_size=0.20, random_state=seed, stratify=label_list
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"   # compensates for uneven label distribution
        ))
    ])

    print("[Fitting model...]")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n" + "="*70)
    print("Classification Report (Macro-F1 is primary metric)")
    print("="*70)
    print(classification_report(y_test, y_pred, labels=LABELS, target_names=LABELS, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS],
                         columns=[f"pred_{l}" for l in LABELS])
    print("Confusion Matrix:\n", cm_df)

    return pipeline, cm_df, (X_test, y_test, y_pred)


def top_terms_per_class(trained_pipeline: Pipeline, top_k: int = 15):
    """Print top TF-IDF terms per class by logistic regression coefficient weight."""
    tfidf = trained_pipeline.named_steps["tfidf"]
    clf = trained_pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_

    print("\n" + "="*70)
    print(f"Top {top_k} Terms Per Topic")
    print("="*70)
    for class_idx, label in enumerate(clf.classes_):
        top_idx = np.argsort(coefs[class_idx])[-top_k:][::-1]
        print(f"\n{label}:")
        for rank, (term, weight) in enumerate(
            zip(feature_names[top_idx], coefs[class_idx][top_idx]), 1
        ):
            print(f"  {rank:2d}. {term:<25} (weight: {weight:>7.3f})")


# ================================================================================
# Severity Helpers (used in critical issues ranking)
# ================================================================================

def rating_str_to_int(r) -> float:
    """Converts '1 star' / '4 stars' → float. Passes through numeric values."""
    if isinstance(r, (int, float)) and not pd.isna(r):
        return float(r)
    if not isinstance(r, str):
        return np.nan
    m = re.match(r"(\d+)", r.strip().lower())
    return float(m.group(1)) if m else np.nan

def rating_to_severity(r: float) -> float:
    """Maps 1–5 star rating to severity weight (1 star = most severe)."""
    if pd.isna(r):
        return np.nan
    return {1: 1.00, 2: 0.75, 3: 0.50, 4: 0.25, 5: 0.00}.get(int(r), np.nan)


# ================================================================================
# run_topic_categorization() — entry point for main.py
# All script logic lives here so this module is safely importable without auto-running.
# ================================================================================

def run_topic_categorization(
    train_csv=TRAIN_CSV,
    inference_csv=INFERENCE_CSV,
    output_csv=OUTPUT_CSV
):
    # === Load & validate training data ===
    df = load_and_filter_reviews(train_csv, text_col="Comment", category_col="Category")

    print(f"\nTotal rows: {len(df)} | Columns: {list(df.columns)}")
    print("\nTopic Label Distribution:")
    label_counts = df["topic_label"].value_counts()
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = 100 * count / len(df) if len(df) > 0 else 0
        print(f"  {label:<25} : {count:>4} ({pct:>5.1f}%)")

    unexpected = set(df["topic_label"].unique()) - set(LABELS)
    if unexpected:
        print(f"WARNING: Unexpected labels found: {unexpected}")
    else:
        print("[OK] All labels valid.")

    # === Train model ===
    if "topic_label" not in df.columns or len(df) == 0:
        print("ERROR: No labeled data available.")
        return None

    model, _, _ = train_and_evaluate_topic_model(
        df["Comment"].astype(str).tolist(), df["topic_label"].tolist()
    )
    top_terms_per_class(model, top_k=15)

    # === Run inference on production data ===
    # Bypasses load_and_filter_reviews() — inference CSV has no Category labels,
    # so the category filter would drop every row.
    df_inference = pd.read_csv(inference_csv)
    df_inference["Comment"] = df_inference["Comment"].astype(str).apply(basic_clean)
    wc = df_inference["Comment"].apply(word_count)

    df_pred = df_inference.copy()
    df_pred["Category"] = None

    valid_mask = wc >= MIN_WORDS
    df_pred.loc[valid_mask, "Category"] = model.predict(
        df_pred.loc[valid_mask, "Comment"].tolist()
    )

    print(f"\nPredicted: {valid_mask.sum()} | Skipped (< {MIN_WORDS} words): {(~valid_mask).sum()}")
    print(df_pred["Category"].value_counts())

    # === Export ===
    df_pred.to_csv(output_csv, index=False)
    print(f"\n[Export] Written to '{output_csv}' ({len(df_pred)} rows)")

    # === Critical issues ranking ===
    # Score = frequency × avg severity; surfaces the most impactful problem areas.
    if "StarRating" in df_pred.columns:
        df_pred["rating_num"] = df_pred["StarRating"].apply(rating_str_to_int)
        df_pred["severity"] = df_pred["rating_num"].apply(rating_to_severity)

        critical = (
            df_pred.dropna(subset=["severity"])
                   .groupby("Category")
                   .agg(freq=("Category", "count"),
                        avg_severity=("severity", "mean"),
                        avg_rating=("rating_num", "mean"))
                   .reset_index()
        )
        critical["critical_score"] = critical["freq"] * critical["avg_severity"]
        critical = critical.sort_values("critical_score", ascending=False)
        print("\nCritical Issues Ranking:\n", critical.to_string(index=False))
    else:
        print("No StarRating column — skipping critical issues ranking.")

    return output_csv


# === allows direct run or clean import by main.py ===
if __name__ == "__main__":
    run_topic_categorization()
