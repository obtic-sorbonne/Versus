"""
Fonctions utilitaires : normalisation, BM25, pondération.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import List
import math


def normalize_embeddings(embeddings: np.ndarray, method: str = "l2") -> np.ndarray:
    """Normalise les embeddings (L2 par défaut)."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if method == "l2":
        return normalize(embeddings, norm='l2', axis=1)
    elif method == "l1":
        return normalize(embeddings, norm='l1', axis=1)
    elif method == "max":
        return normalize(embeddings, norm='max', axis=1)
    return embeddings


class BM25:
    """Implémentation BM25 pour scoring de documents."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc.split()) for doc in corpus) / max(self.corpus_size, 1)

        self.vocab = {}
        self.idf = {}
        self.doc_freqs = []

        self._build_vocab(corpus)
        self._calc_idf()

    def _build_vocab(self, corpus: List[str]):
        for doc in corpus:
            words = doc.lower().split()
            self.doc_freqs.append({})

            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

                word_id = self.vocab[word]
                if word_id not in self.doc_freqs[-1]:
                    self.doc_freqs[-1][word_id] = 0
                self.doc_freqs[-1][word_id] += 1

    def _calc_idf(self):
        df = {}
        for doc_freq in self.doc_freqs:
            for word_id in doc_freq:
                df[word_id] = df.get(word_id, 0) + 1

        for word_id, freq in df.items():
            self.idf[word_id] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_sentence_weights(self, sentences: List[str]) -> np.ndarray:
        """Poids BM25 pour chaque phrase."""
        weights = []
        for sentence in sentences:
            words = sentence.lower().split()
            weight = sum(self.idf.get(self.vocab.get(w, -1), 0) for w in words if w in self.vocab)
            weights.append(weight / max(len(words), 1))
        return np.array(weights)


def compute_tfidf_weights(corpus_texts: List[str], sentences: List[str]) -> np.ndarray:
    """Calcule TF-IDF sur le corpus ENTIER, pas phrase par phrase."""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus_texts)
    tfidf_matrix = vectorizer.transform(sentences)
    weights = np.array(tfidf_matrix.sum(axis=1)).flatten()
    lengths = np.array([max(len(s.split()), 1) for s in sentences])
    return weights / lengths


def compute_bm25_weights(corpus_texts: List[str], sentences: List[str],
                         k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Calcule BM25 sur le corpus ENTIER."""
    bm25 = BM25(corpus_texts, k1=k1, b=b)
    return bm25.get_sentence_weights(sentences)


def weight_embeddings(embeddings: np.ndarray, weights: np.ndarray,
                      strategy: str = "sqrt") -> np.ndarray:
    """Pondère les embeddings selon la stratégie choisie."""
    weights = np.maximum(weights, 0)

    if strategy == "mean":
        weighted = embeddings * weights[:, np.newaxis]
        return np.mean(weighted, axis=0)

    elif strategy == "sqrt":
        sqrt_weights = np.sqrt(weights + 1e-10)
        weighted = embeddings * sqrt_weights[:, np.newaxis]
        return np.mean(weighted, axis=0)

    elif strategy == "weighted_sum":
        weight_sum = np.sum(weights) + 1e-10
        normalized_weights = weights / weight_sum
        weighted = embeddings * normalized_weights[:, np.newaxis]
        return np.sum(weighted, axis=0)

    return np.mean(embeddings, axis=0)
