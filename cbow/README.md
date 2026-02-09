# CBOW Word Embeddings from Scratch (NumPy)

This project implements a **CBOW (Continuous Bag of Words)** word embedding model from scratch using only **NumPy**.

No deep learning libraries (PyTorch / TensorFlow) are used.

The model learns vector representations of words from a text corpus and allows:

- Training CBOW embeddings  
- Visualizing embeddings with PCA  
- Finding similar words using cosine similarity  

---

## Features

- Text preprocessing (regex + NLTK stopwords)  
- Vocabulary building with `<UNK>` token  
- CBOW training with batch gradient descent  
- Custom softmax + cross-entropy loss  
- Word similarity search  
- PCA visualization  
- Fully vectorized forward pass  

---

## Requirements

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn nltk
````

Download NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```

---

## File structure

```
project/
│
├── corpus.txt        # Training text (not included)
├── cbow.py          # Main script
└── README.md
```

---

## How to run

Put your text inside `corpus.txt`, then:

```bash
python cbow.py
```

You will see:

* Training loss per epoch
* Loss curve
* PCA embedding visualization
* Similar words for test inputs

---

## Model parameters

You can modify:

```python
EMBEDDING_DIM = 100
VOCAB_SIZE = 5000
WINDOW = 4
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.05
```

---

## Example output

```
Word: 'man'
  - woman
  - boy
  - father
```

---

## Learning goals

This project was created for educational purposes to understand:

* CBOW architecture
* Backpropagation
* Embedding training
* Cosine similarity
* PCA visualization

---

## Author

Sedra
Machine Learning / NLP practice project

---

