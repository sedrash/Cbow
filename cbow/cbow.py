import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stops = set(stopwords.words("english"))

# === Load and clean text ===
def load_text(filepath):
    """
        Loads text file, cleans it, removes punctuation,
        lowercases everything and removes stopwords.
        Returns list of tokens.
        """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿæœ\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.strip().split()
    tokens = [t for t in tokens if t not in stops]
    return tokens

# === Build vocabulary ===
def build_vocab(tokens, vocab_size=5000):
    """
        Keeps the most frequent words.
        Index 0 is reserved for <UNK>.

        Returns:
        - word_to_index
        - index_to_word
        """

    counter = Counter(tokens)
    most_common = counter.most_common(vocab_size - 1)
    word_to_index = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
    word_to_index['<UNK>'] = 0
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word

# === Encode tokens ===
def encode_tokens(tokens, word_to_index):
    return [word_to_index.get(token, 0) for token in tokens]

# === Generate CBOW data ===
def generate_cbow_data(indexed_tokens, context_window=4):
    """
        Generates CBOW training pairs with:
        - Configurable window size (4 words each side)
        - Center word as target
        - Context words as input
        Returns list of (context, target) tuples
        """
    data = []
    for i in range(context_window, len(indexed_tokens) - context_window):
        context = [indexed_tokens[i + j] for j in range(-context_window, context_window + 1) if j != 0]
        target = indexed_tokens[i]
        data.append((context, target))
    return data

# === Softmax ===
def softmax(x):
    """
       Numerically stable softmax.
       Converts scores to probabilities.
       """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# === Training batch ===
def train_batch(batch_contexts, batch_targets, E, W, b, learning_rate):
    """
        Optimized batch training with:
        - Vectorized operations
        - Parallel gradient computation
        - Weight updates for:
          * Embedding matrix (E)
          * Output weights (W)
          * Biases (b)
        Returns batch loss
        """
    v = np.mean(E[batch_contexts], axis=1)
    scores = np.dot(v, W) + b
    probs = softmax(scores)
    batch_size = batch_targets.shape[0]
    loss = -np.log(probs[np.arange(batch_size), batch_targets] + 1e-10).mean()

    dscores = probs
    dscores[np.arange(batch_size), batch_targets] -= 1
    dscores /= batch_size

    dW = np.dot(v.T, dscores)
    db = np.sum(dscores, axis=0)
    dv = np.dot(dscores, W.T)

    for i in range(batch_contexts.shape[1]):
        for j in range(batch_size):
            E[batch_contexts[j, i]] -= learning_rate * dv[j] / batch_contexts.shape[1]

    W -= learning_rate * dW
    b -= learning_rate * db

    return loss

# === Visualize embeddings ===
def visualize_embeddings(E, index_to_word, num_words=50):
    words = list(index_to_word.keys())[:num_words]
    labels = [index_to_word[i] for i in words]
    vectors = E[words]
    reduced = PCA(n_components=2).fit_transform(vectors)
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, label, fontsize=8)
    plt.title("Word Embedding Visualization (PCA)")
    plt.grid(True)
    plt.show()

# === Find neighbors with filtering ===
def find_neighbors(word, word_to_index, index_to_word, embeddings, k=5, min_index=10, max_index=4000):
    """
        Finds similar words using:
        - Cosine similarity
        - Normalized embeddings
        - Configurable result count
        - Vocabulary range filtering
        """
    if word not in word_to_index:
        return []
    idx = word_to_index[word]
    norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    query_vec = norm_embeddings[idx].reshape(1, -1)
    sim = cosine_similarity(query_vec, norm_embeddings)[0]
    candidates = [i for i in range(len(index_to_word)) if i != idx and min_index <= i < max_index]
    candidates.sort(key=lambda i: sim[i], reverse=True)
    return [(index_to_word[i], sim[i]) for i in candidates[:k]]

# === Parameters ===
corpus_path = "corpus.txt"
EMBEDDING_DIM = 100
VOCAB_SIZE = 5000
LEARNING_RATE = 0.05
EPOCHS = 10
WINDOW = 4
BATCH_SIZE = 64

# === Preprocessing ===
tokens = load_text(corpus_path)[:300000]
word_to_index, index_to_word = build_vocab(tokens, vocab_size=VOCAB_SIZE)
indexed = encode_tokens(tokens, word_to_index)
data = generate_cbow_data(indexed, context_window=WINDOW)

contexts = np.array([x[0] for x in data])
targets = np.array([x[1] for x in data])

# === Initialization ===
E = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM) * 0.01
W = np.random.randn(EMBEDDING_DIM, VOCAB_SIZE) * 0.01
b = np.zeros(VOCAB_SIZE)
losses = []

# === Training ===
print("\n=== TRAINING CBOW MODEL (OPTIMIZED) ===")
for epoch in range(EPOCHS):
    total_loss = 0
    perm = np.random.permutation(len(data))
    for i in range(0, len(data), BATCH_SIZE):
        batch_idx = perm[i:i + BATCH_SIZE]
        batch_context = contexts[batch_idx]
        batch_target = targets[batch_idx]
        loss = train_batch(batch_context, batch_target, E, W, b, LEARNING_RATE)
        total_loss += loss
    avg_loss = total_loss / (len(data) / BATCH_SIZE)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.2f}")
    losses.append(avg_loss)

# === Loss plot ===
plt.plot(losses)
plt.title("Training Loss (Optimized CBOW)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# === Visualization ===
visualize_embeddings(E, index_to_word, num_words=500)

# === Test words ===
test_words = [
    "man", "woman", "love", "money", "life", "death",
    "young", "old", "happy", "sad", "day", "night",
    "house", "rich", "dream", "father", "mother"
]

print("\n=== WORD SIMILARITY TESTS ===")
for word in test_words:
    if word in word_to_index:
        print(f"\nWord: '{word}' — Top 5 similar words:")
        for neighbor, score in find_neighbors(word, word_to_index, index_to_word, E, k=5):
            print(f"  - {neighbor} (similarity: {score:.3f})")
    else:
        print(f"\nWord: '{word}' not in vocabulary.")
