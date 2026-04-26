# ================================================================================
# FixYourFood – Topic Categorization (TF-IDF + Logistic Regression)
# ================================================================================
#
# **GOAL:** Assign each McDonald's review ONE topic label using supervised ML:
#   - FOOD_QUALITY (burger, fries, nuggets quality, temperature, staleness, etc.)
#   - STAFF_SERVICE (cashier, manager, employee behavior, politeness, speed)
#   - TIMELINESS_SPEED (wait times, order fulfillment time, drive-thru speed)
#   - CLEANLINESS (bathrooms, tables, floors, hair, trash, overall hygiene)
#   - ENVIRONMENT_ATMOSPHERE (noise, seating, ambiance, PlayPlace, crowding)
#
# **MODEL:** TF-IDF Vectorizer + Logistic Regression (balanced class weights)
# **VALIDATION:** Stratified 80/20 holdout. Metrics: Macro-F1 + per-class F1
# **DATA FILTER:** Only reviews with 5+ words are included.
#
# **TEAM NOTES:**
#  - Sentiment Analysis team: We output pred_topic; you can filter by rating.
#  - Other teams: Join on review text or index for cross-analysis.
#  - GitHub: Please comment on PRs with feature requests or data issues.
# ================================================================================

# ================================================================================
# CELL 1: Imports + Constants
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
# These are the 6 categories I saw in the data. All must be mapped to one of these.
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
# Handles slight variations (spacing, case) from data collection.
# ***** Both "Order accuracy" and "Customization error" map to CUSTOMIZATION_ERROR
#       because they represent order fulfillment/customization issues.
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

# -------- CONFIGURATION CONSTANTS --------
MIN_WORDS = 5  # Minimum word count to include a review
RANDOM_SEED = 42  # For reproducibility across runs

# ================================================================================
# CELL 2: Helper Functions (Text Cleaning + Filtering)
# ================================================================================
# These functions handle data validation, cleaning, and filtering.
# helps with the heavy lifting done by TF-IDF vectorizer.

def word_count(text: str) -> int:
    """Count words in text string. Returns 0 for non-strings."""
    if not isinstance(text, str):
        return 0
    return len([w for w in text.strip().split() if w])

def basic_clean(text: str) -> str:
    """
    text preprocessing (normalization).
      - Convert to string if needed
      - Strip leading/trailing whitespace
      - Collapse multiple spaces into single spaces
       """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text

def normalize_category(category_str: str) -> str:
    """
    Map raw CSV category to standardized label.
    Args:
        category_str: Raw category from CSV ("Food quality")
    Returns:
        Standardized label ("FOOD_QUALITY") or None if unmapped.
        ***Reviews with unmapped categories are dropped during filtering.
    """
    if not isinstance(category_str, str):
        return None
    category_str = category_str.strip()
    return CATEGORY_MAPPING.get(category_str, None)

def load_and_filter_reviews(
    csv_path: str,
    text_col: str = "Comment",
    category_col: str = "Category"
) -> pd.DataFrame:
    """
    Load CSV, clean text, filter short reviews, and map categories.
    Args:
        csv_path: Path to the labeled CSV file used for training (e.g. TestSampleData.csv)
        text_col: Column name containing review text (default: "Comment")
        category_col: Column name containing category labels (default: "Category")
    Returns:
        Filtered DataFrame with:
          - Cleaned text in text_col
          - Normalized categories in 'topic_label' column
          - Rows with < {MIN_WORDS} words removed
          - Duplicates by review text removed
          - Unmapped categories removed
        ValueError: If required columns not found in CSV
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns exist
    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}'. Found: {list(df.columns)}")
    if category_col not in df.columns:
        raise ValueError(f"Expected column '{category_col}'. Found: {list(df.columns)}")
    
    # Clean review text
    df[text_col] = df[text_col].astype(str).apply(basic_clean)
    df["word_count"] = df[text_col].apply(word_count)
    
    # Map raw categories to standardized labels
    df["topic_label"] = df[category_col].apply(normalize_category)
    
    # Filter: keep only valid (mapped) categories and sufficient word count
    initial_count = len(df)
    df = df[df["topic_label"].notna()].copy()  # Drop unmapped categories
    df = df[df["word_count"] >= MIN_WORDS].copy()  # Drop short reviews
    
    # Remove exact duplicate review texts (keep the first occurrence)
    df = df.drop_duplicates(subset=[text_col], keep="first").copy()
    
    # Reset index for clean sequential IDs
    df.reset_index(drop=True, inplace=True)
    
    # Report filtering
    final_count = len(df)
    print(f"[Data Filtering Summary]")
    print(f"  Initial rows: {initial_count}")
    print(f"  After mapping + filtering: {final_count}")
    print(f"  Rows removed: {initial_count - final_count}")
    print()
    
    return df

# ================================================================================
# CELL 3: Load & Inspect Data
# ================================================================================
# Two data sources are used:
#   - TRAIN_CSV (TestSampleData.csv): pre-labeled reviews used to train the model
#   - INFERENCE_CSV (mcdonaldsfoodreviews.csv): real production reviews with no labels
# The model learns from TRAIN_CSV and applies predictions to INFERENCE_CSV.

TRAIN_CSV = "TestSampleData.csv"        # Labeled data → trains the model
INFERENCE_CSV = "mcdonaldsfoodreviews.csv"  # Production data → receives predictions

# Load with standard column names for our dataset
df = load_and_filter_reviews(
    csv_path=TRAIN_CSV,
    text_col="Comment",        # Review text column
    category_col="Category"    # Topic category column
)

print("First 5 reviews (after filtering):")
print(df[["Comment", "topic_label"]].head())
print()

# ================================================================================
# CELL 4: Data Sanity Checks
# ================================================================================
# Verify data shape, column names, and label distribution.

print("=" * 70)
print("Data Shape & Structure")
print("=" * 70)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Show topic label distribution (to help with class balance)
print("Topic Label Distribution:")
label_counts = df["topic_label"].value_counts()
for label in LABELS:
    count = label_counts.get(label, 0)
    pct = 100 * count / len(df) if len(df) > 0 else 0
    print(f"  {label:<25} : {count:>4} reviews ({pct:>5.1f}%)")
print()

# Verify all labels are in our original set
unique_labels = set(df['topic_label'].unique())
if not unique_labels.issubset(set(LABELS)):
    print(f'WARNING: Found unexpected labels: {unique_labels - set(LABELS)}')
else:
    print('[OK] All labels are valid and in LABELS column.')
print()

# ================================================================================
# CELL 5: Using Pre-Labeled Data
# ================================================================================
# The load_and_filter_reviews() function above handles mapping those to our
# standardized LABELS. #
# Rule: Each review gets EXACTLY ONE label (multi-class),
#       choosing the MOST PROMINENT or SERIOUS topic.


# ================================================================================
# CELL 6: Training Data Preparation (If Using Pre-Labeled Subset)
# ================================================================================
# I wrote this part just in case we have any unlabled reviews that need manual labeling.
# I can uncomment and use if we have unlabeled reviews
# def create_labeling_subset(
#     df: pd.DataFrame,
#     n: int = 600,
#     seed: int = RANDOM_SEED
# ) -> pd.DataFrame:
#     """
#     Create random subset of reviews for manual labeling.
#     
#     Args:
#         df: Unlabeled review dataframe
#         n: Number of reviews to sample
#         seed: Random seed for reproducibility
#     
#     Returns:
#         Subset with empty topic_label column ready for manual filling.
#     """
#     n = min(n, len(df))
#     subset = df.sample(n=n, random_state=seed).copy()
#     subset["topic_label"] = ""  # to be filled manually
# 
#     # Keep columns useful for labeling decision
#     keep_cols = []
#     for c in ["Comment", "StarRating", "FoodQuality", "topic_label"]:
#         if c in subset.columns:
#             keep_cols.append(c)
#     
#     subset = subset[keep_cols].copy()
#     subset.reset_index(drop=True, inplace=True)
#     return subset
#
# # Example: Create and export for labeling
# # label_subset = create_labeling_subset(df, n=600)
# # label_subset.to_csv("labeling_subset_to_fill.csv", index=False)
# # print("Export completed. Fill topic_label column with one of:")
# # for label in LABELS:
# #     print(f"  - {label}")


# ================================================================================
# PART 2: MODEL TRAINING & EVALUATION
# ================================================================================
# Pipeline: TF-IDF Vectorizer + Logistic Regression 
# Splits: Stratified 80/20 train/test 
# Metrics: Macro-F1 (overall), Per-class F1 (per category)
# ================================================================================
# CELL 7: Model Training & Evaluation Function
def train_and_evaluate_topic_model(
    text_list,
    label_list,
    seed: int = RANDOM_SEED
):
    """
    Train TF-IDF + Logistic Regression topic classifier.
    Evaluate on 80/20 split.
    Args:
        text_list: List of review text strings
        label_list: List of topic labels
        seed: Random seed for train/test split reproducibility
    Returns:
        (pipeline, confusion_matrix_df, (X_test, y_test, y_pred))
    Side Effects:
        Prints classification report, confusion matrix, and metrics.
    """
    # Verify we have matching data
    assert len(text_list) == len(label_list), "Text and label lists must have same length"
    assert len(text_list) > 0, "Empty data provided"
    
    # Verify all labels are match
    invalid_labels = set(label_list) - set(LABELS)
    if invalid_labels:
        raise ValueError(f"Found invalid labels not in LABELS: {invalid_labels}")
    
    X = text_list
    y = label_list
    
    print("\n" + "="*70)
    print("Training Topic Classification Model")
    print("="*70)
    
    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=seed,
        stratify=y  # <-- ensures proportional class distribution in train/test
    )
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Build ML pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),            # Use unigrams and bigrams
            min_df=2,                      # Ignore terms appearing < 2 times
            max_df=0.95,                   # Ignore terms appearing in > 95% of docs
            stop_words="english"          # Remove common English words
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"        # Handle class imbalance
        ))
    ])
    
    # Train the pipeline
    print("[Fitting model...]")
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # ===== Evaluation Metrics =====
    print("\n" + "="*70)
    print("Classification Report (Macro-F1 is primary metric)")
    print("="*70)
    print(classification_report(
        y_test, y_pred,
        labels=LABELS,
        target_names=LABELS,
        digits=3
    ))
    
    print("\n" + "="*70)
    print("Confusion Matrix (rows=true labels, cols=predictions)")
    print("="*70)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS]
    )
    print(cm_df)
    
    return pipeline, cm_df, (X_test, y_test, y_pred)

# ================================================================================
# CELL 8: Train the Model (Using Loaded Data)
# ================================================================================
# If df already has topic_label (from load_and_filter_reviews):

if "topic_label" in df.columns and len(df) > 0:
    print("[Using pre-labeled data for training]")
    text_list = df["Comment"].astype(str).tolist()
    label_list = df["topic_label"].tolist()
    
    model, cm_df, test_pack = train_and_evaluate_topic_model(text_list, label_list)
else:
    print("ERROR: No labeled data found. Ensure df has 'topic_label' column.")
    model, cm_df, test_pack = None, None, None

# ================================================================================
# CELL 9: Feature Importance Analysis (Top Terms Per Topic)
# ================================================================================
# Analyze which TF-IDF terms are most predictive for each topic.
# This will be useful for understanding model decisions and finding key terms.

def top_terms_per_class(trained_pipeline: Pipeline, top_k: int = 15):
    """
    Extract and display top TF-IDF terms for each topic class.
    Args:
        trained_pipeline: Fitted Pipeline with TfidfVectorizer + LogisticRegression
        top_k: Number of top terms to show per class
    Uses logistic regression coefficients to rank feature importance.
    High positive coefficient = strong signal for that class
    """
    tfidf = trained_pipeline.named_steps["tfidf"]
    clf = trained_pipeline.named_steps["clf"]
    
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_
    
    print("\n" + "="*70)
    print(f"Top {top_k} Terms Per Topic (by LogisticRegression coefficient)")
    print("="*70)
    
    for class_idx, label in enumerate(clf.classes_):
        top_idx = np.argsort(coefs[class_idx])[-top_k:][::-1]
        print(f"\n{label}:")
        for rank, (term, weight) in enumerate(zip(feature_names[top_idx], coefs[class_idx][top_idx]), 1):
            print(f"  {rank:2d}. {term:<25} (weight: {weight:>7.3f})")

if model:
    top_terms_per_class(model, top_k=15)


# ================================================================================
# PART 3: PREDICTIONS & PRELIMINARY ANALYTICS
# ================================================================================
# Use trained model to:
#  1) Predict topic for all reviews
#  2) Analyze topic trends over time (if we have review_time available)
#  3) Rank topics by severity (if ratings available)
# ====================================================================



# ================================================================================
# CELL 10: Time Analysis Helpers (Review Age & Time Buckets)
# ================================================================================

def review_time_to_days(s: str) -> float:
    """
    Convert relative time string to approximate days ago.
    Examples: "5 days ago" -> 5, "3 months ago" -> 90, "a year ago" -> 365
    Args:
        s: Time string from CSV
    Returns:
        Approximate number of days, or np.nan if unparseable
    ***all months = 30 days).
    For precise timestamps, ask data team for DateTime column.
    """
    if not isinstance(s, str):
        return np.nan
    s = s.strip().lower()
    
    # Handle spelled-out units
    if s in ["a day ago", "1 day ago"]:
        return 1.0
    if s in ["a week ago", "1 week ago"]:
        return 7.0
    if s in ["a month ago", "1 month ago"]:
        return 30.0
    if s in ["a year ago", "1 year ago"]:
        return 365.0
    
    # Handle "N unit(s) ago" pattern
    m = re.match(r"(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago", s)
    if not m:
        return np.nan
    
    num = int(m.group(1))
    unit = m.group(2).lower()
    
    # Convert to days
    if "day" in unit:
        return float(num)
    elif "week" in unit:
        return float(num * 7)
    elif "month" in unit:
        return float(num * 30)  # Approximation
    elif "year" in unit:
        return float(num * 365)  # Approximation
    
    return np.nan

# ===========================================================================
# Cell 11 — Severity mapping from rating strings (e.g., “1 star”, “4 stars”)
# ===========================================================================
def rating_str_to_int(r) -> float:
    """
    Converts '1 star' / '4 stars' -> int rating.
    If already numeric, keep it.
    """
    if isinstance(r, (int, float)) and not pd.isna(r):
        return float(r)
    if not isinstance(r, str):
        return np.nan
    r = r.strip().lower()
    m = re.match(r"(\d+)", r)
    return float(m.group(1)) if m else np.nan

def rating_to_severity(r: float) -> float:
    """
    Simple severity weight:
    1 -> 1.00
    2 -> 0.75
    3 -> 0.50
    4 -> 0.25
    5 -> 0.00
    """
    if pd.isna(r):
        return np.nan
    mapping = {1: 1.00, 2: 0.75, 3: 0.50, 4: 0.25, 5: 0.00}
    return mapping.get(int(r), np.nan)

# ================================================================================
# CELL 12: Generate Predictions on Full Dataset
# ================================================================================

if model:
    print("\n" + "="*70)
    print("Generating Predictions on Full Dataset")
    print("="*70)
    
    # Load production data directly — bypasses load_and_filter_reviews() because
    # INFERENCE_CSV has no Category labels to map, so filtering would drop all rows.
    # We only need clean text; the model handles everything else.
    df_inference = pd.read_csv(INFERENCE_CSV)
    df_inference["Comment"] = df_inference["Comment"].astype(str).apply(basic_clean)  # normalize text same as training

    # Copy data and add predictions
    df_pred = df_inference.copy()
    df_pred["pred_topic"] = model.predict(df_pred["Comment"].tolist())
    
    # Verify predictions exist
    print(f"Predictions generated for {len(df_pred)} reviews")
    print(f"Topic distribution in predictions:")
    print(df_pred["pred_topic"].value_counts())
    print()
else:
    print("No model available; skipping predictions.")
    df_pred = None

# ================================================================================
# CELL 12b: Export Predictions to CSV
# ================================================================================
# Writes pred_topic back as a new column in a separate output CSV.
# Raw input CSV is intentionally preserved; teammates write their own output CSVs.
# main.py will merge all team output files for final cross-analysis.

OUTPUT_CSV = "mcdonaldsfoodreviews_predicted.csv"  # New file — raw source CSV is never overwritten

if df_pred is not None:
    df_pred.to_csv(OUTPUT_CSV, index=False)
    print(f"[Export] Predictions written to '{OUTPUT_CSV}' ({len(df_pred)} rows, {len(df_pred.columns)} columns)")
    print(f"         Columns: {list(df_pred.columns)}")
else:
    print("[Export] No predictions available; skipping CSV export.")


# ================================================================================
# CELL 13: Critical Issues Analysis (Topics by Severity)
# ================================================================================
# Rank topics by urgency using predicted topic + rating severity.
# Formula: CriticalScore = (# of reviews) × (avg severity weight)
# Rationale: A topic is more critical if it appears frequently AND correlates with low ratings.

if df_pred is not None and "StarRating" in df_pred.columns:
    print("\n" + "="*70)
    print("Critical Issues Ranking (Topics by Severity)")
    print("="*70)
    
    # Convert rating to numeric and severity
    df_pred["rating_num"] = df_pred["StarRating"].apply(rating_str_to_int)
    df_pred["severity"] = df_pred["rating_num"].apply(rating_to_severity)
    
    # Compute critical score per topic
    critical = (
        df_pred.dropna(subset=["severity"])
              .groupby("pred_topic")
              .agg(
                  freq=("pred_topic", "count"),
                  avg_severity=("severity", "mean"),
                  avg_rating=("rating_num", "mean")
              )
              .reset_index()
    )
    critical["critical_score"] = critical["freq"] * critical["avg_severity"]
    critical = critical.sort_values("critical_score", ascending=False)
    
    print("\nRanked by Critical Score (frequency × avg_severity):")
    print(critical.to_string(index=False))
    print()
elif df_pred is not None:
    print("No StarRating column found; skipping critical points analysis.")
else:
    print("No predictions available; skipping critical analysis.")