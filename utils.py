"""
Fonctions utilitaires : normalisation, BM25, pondération, fusion segments.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import List, Tuple
import math


def normalize_embeddings(embeddings: np.ndarray, method: str = "l2") -> np.ndarray:
    """Normalise les embeddings (L2 par défaut - CRITIQUE)."""
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
        self.vocab = {}
        self.idf = {}
        self.doc_freqs = []

        if not corpus:
            self.avgdl = 0.0
            return

        self.avgdl = sum(len(doc.split()) for doc in corpus) / self.corpus_size
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
        if not self.vocab:
            return np.ones(len(sentences))
        weights = []
        for sentence in sentences:
            words = sentence.lower().split()
            weight = sum(self.idf.get(self.vocab.get(w, -1), 0) for w in words if w in self.vocab)
            weights.append(weight / max(len(words), 1))
        return np.array(weights)


def compute_tfidf_weights(corpus_texts: List[str], sentences: List[str]) -> np.ndarray:
    """
    CRITIQUE: Calcule TF-IDF sur le corpus ENTIER, pas phrase par phrase.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus_texts)
    tfidf_matrix = vectorizer.transform(sentences)
    weights = np.array(tfidf_matrix.sum(axis=1)).flatten()
    lengths = np.array([max(len(s.split()), 1) for s in sentences])
    return weights / lengths


def compute_bm25_weights(corpus_texts: List[str], sentences: List[str], 
                        k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """
    CRITIQUE: Calcule BM25 sur le corpus ENTIER.
    """
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


def adaptive_threshold(base_threshold: float, segment_length: int, 
                      alpha: float = 0.05) -> float:
    """Seuil adaptatif selon longueur segment."""
    adjustment = alpha * math.log(max(segment_length, 1))
    return max(0.5, base_threshold - adjustment)


def merge_overlapping_segments(segments: List[Tuple[int, int]], 
                               max_gap: int = 5) -> List[Tuple[int, int]]:
    """Fusionne les segments qui se chevauchent ou sont proches."""
    if not segments:
        return []
    
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        last = merged[-1]
        if current[0] <= last[1] + max_gap:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def classify_match_type(similarity: float) -> Tuple[str, str]:
    """
    Classifie le type de correspondance selon le score.
    Retourne (type, couleur_css).
    """
    if similarity >= 0.95:
        return "exact", "rgba(16, 185, 129, 0.4)"
    elif similarity >= 0.85:
        return "overlap", "rgba(59, 130, 246, 0.35)"
    elif similarity >= 0.75:
        return "reformulation", "rgba(245, 158, 11, 0.4)"
    else:
        return "semantic", "rgba(147, 51, 234, 0.3)"


def get_match_type_label(match_type: str) -> str:
    """Label français pour le type de match."""
    labels = {
        "exact": "Exact",
        "overlap": "Chevauchement",
        "reformulation": "Reformulation",
        "semantic": "Sémantique"
    }
    return labels.get(match_type, match_type)
