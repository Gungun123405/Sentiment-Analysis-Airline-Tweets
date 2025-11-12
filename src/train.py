# ==============================================
# train.py — Final Optimized Deep BiLSTM + Attention
# ==============================================

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- FIX IMPORT PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.deep_bilstm_attention import build_bilstm_attention_model

# ===========================================
# CONFIGURATION
# ===========================================

DATASET_PATH = os.path.join("data", "processed", "dataset.pkl")
TOKENIZER_PATH = os.path.join("data", "processed", "tokenizer.pkl")
EMBEDDING_PATH = os.path.join("data", "glove", "glove.6B.200d.txt")
REPORTS_DIR = "reports"
MODEL_PATH = os.path.join("models", "best_optimized_bilstm.h5")

MAX_LEN = 60
EMBED_DIM = 200
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 3e-4

# ===========================================
# LOAD DATASET
# ===========================================

print("📦 Loading preprocessed dataset...")
with open(DATASET_PATH, "rb") as f:
    X, y = pickle.load(f)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

num_words = len(tokenizer.word_index) + 1
num_classes = y.shape[1]

print(f"✅ Loaded dataset: X={X.shape}, y={y.shape}, vocab_size={num_words}")

# ===========================================
# TRAIN-VALIDATION SPLIT
# ===========================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
)
print(f"📊 Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# ===========================================
# LOAD GLOVE EMBEDDINGS
# ===========================================

def load_glove_embeddings(filepath, word_index, embed_dim=200):
    print(f"\n🔤 Loading GloVe embeddings ({embed_dim}d)...")
    embeddings_index = {}
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print(f"✅ Loaded {len(embeddings_index)} embeddings.")
    return embedding_matrix


embedding_matrix = load_glove_embeddings(EMBEDDING_PATH, tokenizer.word_index, EMBED_DIM)

# ===========================================
# BUILD MODEL
# ===========================================

print("\n🧠 Building Deep BiLSTM + Attention model...")
model = build_bilstm_attention_model(
    embedding_matrix=embedding_matrix,
    max_len=MAX_LEN,
    num_words=num_words,
    embed_dim=EMBED_DIM,
    num_classes=num_classes,
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ===========================================
# TRAINING CONFIGURATION
# ===========================================

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
]

# ===========================================
# TRAIN MODEL
# ===========================================

print("\n🚀 Starting model training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

print("\n✅ Training complete! Best model saved at:", MODEL_PATH)

# ===========================================
# PLOT TRAINING CURVES
# ===========================================

os.makedirs(REPORTS_DIR, exist_ok=True)
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "training_curves.png"))
plt.show()

print("\n📊 Training curves saved at:", os.path.join(REPORTS_DIR, "training_curves.png"))
