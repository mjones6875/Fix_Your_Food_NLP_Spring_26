# ================================================================================
# FoodSentiment – CNN Sentiment Classifier (Star Rating Prediction)
# ================================================================================
# Fine-tunes a TextCNN on GloVe embeddings to predict star ratings (1–5)
# for McDonald's reviews.
#
# Trains on mcdonaldsfoodreviews.csv (reviews with existing StarRating values).
# Writes pred_sentiment column (integer 1–5) to mcdonaldsfoodreviews_sentiment.csv.
# ================================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras import layers, Model
import os

# File paths — used as defaults in run_sentiment_prediction()
INFERENCE_CSV = "mcdonaldsfoodreviews.csv"
GLOVE_PATH = "glove.6B.100d.txt"
OUTPUT_CSV = "mcdonaldsfoodreviews_sentiment.csv"  # never overwrites source data


# ================================================================================
# GloVe Loader
# ================================================================================

def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# ================================================================================
# Model Architecture
# ================================================================================

def build_improved_textcnn(vocab_size, max_len, embedding_dim=100,
                            embedding_matrix=None, num_classes=5,
                            kernel_sizes=[2, 3, 4, 5], filters_per_kernel=128,
                            dense_units=[256, 128], dropout_rate=0.5,
                            spatial_dropout_rate=0.3, lr=1e-3):
    inputs = layers.Input(shape=(max_len,), dtype='int32')
    if embedding_matrix is not None:
        embedding = layers.Embedding(vocab_size, embedding_dim,
                                     weights=[embedding_matrix],
                                     trainable=True, name='embedding')(inputs)
    else:
        embedding = layers.Embedding(vocab_size, embedding_dim,
                                     name='embedding')(inputs)
    embedding = layers.SpatialDropout1D(spatial_dropout_rate)(embedding)

    pooled_outputs = []
    for k in kernel_sizes:
        conv = layers.Conv1D(filters_per_kernel, k, padding='same',
                             activation='relu', name=f'conv_{k}')(embedding)
        max_pool = layers.GlobalMaxPooling1D(name=f'maxpool_{k}')(conv)
        avg_pool = layers.GlobalAveragePooling1D(name=f'avgpool_{k}')(conv)
        pooled = layers.Concatenate(name=f'pool_{k}')([max_pool, avg_pool])
        pooled_outputs.append(pooled)

    concat = layers.Concatenate()(pooled_outputs)
    x = layers.BatchNormalization()(concat)
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ================================================================================
# Data Loading
# ================================================================================

def load_reviews(file_path, star_col='StarRating', text_col='Comment',
                 sheet_name=None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path, usecols=[star_col, text_col])
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(file_path,
                           sheet_name=0 if sheet_name is None else sheet_name,
                           usecols=[star_col, text_col])
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx/.xls")

    df.dropna(subset=[star_col, text_col], inplace=True)

    # str.extract pulls the digit from strings like "1 star" / "2 stars" before
    # converting — the original .astype(int) crashed on these string values
    df[star_col] = df[star_col].astype(str).str.extract(r'(\d+)', expand=False).astype(float)

    df = df[df[star_col].between(1, 5)]
    return df


# ================================================================================
# Preprocessing
# ================================================================================

def preprocess_data(df, star_col, text_col, max_len, max_words=20000,
                    test_size=0.2, val_size=0.1):
    texts = df[text_col].astype(str).tolist()
    ratings = df[star_col].values

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>', lower=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    y = tf.keras.utils.to_categorical(ratings - 1, num_classes=5)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=42
    )
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer


# ================================================================================
# Analytics (separated from training so import doesn't trigger plotting)
# ================================================================================

def run_analytics(df, X_train, X_val, X_test, y_test, model, history):
    """Display training curves, classification report, and confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    class_labels = ['1★', '2★', '3★', '4★', '5★']

    print("\nDATASET STATISTICS")
    print(f"Total reviews loaded: {len(df)}")
    print("Class distribution (original):")
    class_counts = df['StarRating'].value_counts().sort_index()
    for star, count in class_counts.items():
        print(f"  {int(star)} stars: {count:5d}  ({100 * count / len(df):.1f}%)")
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\nTRAINING HISTORY")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Final training accuracy:   {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best validation accuracy:  {max(history.history['val_accuracy']):.4f} "
          f"(epoch {np.argmax(history.history['val_accuracy']) + 1})")

    print("\nTEST SET EVALUATION")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("\nPER-CLASS CONFIDENCE")
    for i, label in enumerate(class_labels):
        mask = y_true == i
        if np.sum(mask) > 0:
            probs = y_pred_probs[mask, i]
            print(f"{label}: mean={np.mean(probs):.4f}, std={np.std(probs):.4f}, "
                  f"min={np.min(probs):.4f}")


# ================================================================================
# main() — trains and evaluates the model
# ================================================================================

def main(file_path=INFERENCE_CSV, glove_path=GLOVE_PATH):
    # GloVe check moved here from module level — raises instead of silently printing
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            f"GloVe file not found: '{glove_path}'. "
            "Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ and extract it."
        )

    STAR_COL = 'StarRating'
    TEXT_COL = 'Comment'
    MAX_LEN = 100
    MAX_WORDS = 20000
    EMBEDDING_DIM = 100
    BATCH_SIZE = 64
    EPOCHS = 50

    df = load_reviews(file_path, star_col=STAR_COL, text_col=TEXT_COL)
    print(f"Loaded {len(df)} reviews.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer = \
        preprocess_data(df, STAR_COL, TEXT_COL, MAX_LEN, max_words=MAX_WORDS)

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    embedding_matrix = load_glove_embeddings(glove_path, tokenizer.word_index, EMBEDDING_DIM)
    model = build_improved_textcnn(
        vocab_size=vocab_size, max_len=MAX_LEN, embedding_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix, num_classes=5
    )
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # tokenizer and MAX_LEN returned so run_sentiment_prediction() can preprocess new data
    return df, X_train, y_train, X_val, y_val, X_test, y_test, model, history, tokenizer, MAX_LEN


# ================================================================================
# run_sentiment_prediction() — entry point for main.py
# Trains the model then runs inference on the full dataset, writing pred_sentiment.
# ================================================================================

def run_sentiment_prediction(
    inference_csv=INFERENCE_CSV,
    glove_path=GLOVE_PATH,
    output_csv=OUTPUT_CSV
):
    # Train and evaluate
    df, X_train, _, X_val, _, X_test, y_test, model, history, tokenizer, MAX_LEN = \
        main(file_path=inference_csv, glove_path=glove_path)

    # Load full dataset for inference (unfiltered — includes blank reviews)
    df_full = pd.read_csv(inference_csv)
    texts = df_full["Comment"].fillna("").astype(str).tolist()

    sequences = tokenizer.texts_to_sequences(texts)
    X_all = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # argmax gives 0–4; shift to 1–5 to match original star rating scale
    y_pred = np.argmax(model.predict(X_all), axis=1) + 1
    df_full["pred_sentiment"] = y_pred

    df_full.to_csv(output_csv, index=False)
    print(f"[Export] Written to '{output_csv}' ({len(df_full)} rows)")

    return output_csv


# === allows direct run or clean import by main.py ===
if __name__ == "__main__":
    run_sentiment_prediction()
