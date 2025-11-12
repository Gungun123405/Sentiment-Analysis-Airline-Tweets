# ==============================================
# evaluate.py — Robust evaluation for Deep BiLSTM + Attention
# ==============================================

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# --- Ensure project root is in import path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.deep_bilstm_attention import build_bilstm_attention_model  # Build function, not just layer

# ===========================================
# CONFIGURATION
# ===========================================

DATA_PATH = os.path.join("data", "processed", "dataset.pkl")
TOKENIZER_PATH = os.path.join("data", "processed", "tokenizer.pkl")
MODEL_PATH = os.path.join("models", "best_optimized_bilstm.h5")  # This will now be WEIGHTS
REPORTS_DIR = "reports"
EMBEDDING_PATH = os.path.join("data", "glove", "glove.6B.200d.txt")
MAX_LEN = 60
EMBED_DIM = 200

# ===========================================
# LOAD DATA
# ===========================================

print("📦 Loading preprocessed dataset...")
with open(DATA_PATH, "rb") as f:
    X, y = pickle.load(f)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

num_words = len(tokenizer.word_index) + 1
num_classes = y.shape[1]

print(f"✅ Dataset loaded: X={X.shape}, y={y.shape}, vocab_size={num_words}")

# ===========================================
# LOAD EMBEDDINGS
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
# REBUILD MODEL ARCHITECTURE
# ===========================================

print("\n🧠 Rebuilding model architecture for evaluation...")
model = build_bilstm_attention_model(
    embedding_matrix=embedding_matrix,
    max_len=MAX_LEN,
    num_words=num_words,
    embed_dim=EMBED_DIM,
    num_classes=num_classes,
)
model.load_weights(MODEL_PATH)

# 🧩 Compile before evaluation
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ Model architecture rebuilt, weights loaded, and compiled!\n")

# ===========================================
# EVALUATE MODEL
# ===========================================

print("⚙️ Evaluating model...")
loss, accuracy = model.evaluate(X, y, verbose=1)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_pred = model.predict(X)
y_true_labels = np.argmax(y, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# ===========================================
# METRICS + VISUALS
# ===========================================

target_names = ["Negative", "Neutral", "Positive"]
report = classification_report(y_true_labels, y_pred_labels, target_names=target_names)
print("\n📊 Classification Report:\n")
print(report)

# Save report
os.makedirs(REPORTS_DIR, exist_ok=True)
with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
plt.show()

# F1 Scores Bar Plot
prec, rec, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average=None)
plt.figure(figsize=(6, 4))
sns.barplot(x=target_names, y=f1, palette="crest")
plt.title("F1 Scores by Sentiment Class")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "f1_scores.png"))
plt.show()

print(f"\n✅ Evaluation complete! Results saved in '{REPORTS_DIR}/'")
