"""
Corpus : gestion avec calcul des poids sur l'ENSEMBLE du corpus.
"""

from Global_stuff import Global_stuff
from Document import Document
from os import walk, path
from copy import copy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from utils import compute_tfidf_weights, compute_bm25_weights, normalize_embeddings, BM25
from Config import ComparisonConfig


class Corpus:
    """Gestion du corpus avec calcul global des poids."""

    def __init__(self, path_dir: str = None, config: ComparisonConfig = None):
        if path_dir:
            documents_paths = []
            for (dirpath, _, filenames) in walk(path_dir):
                documents_paths.extend([
                    path.join(dirpath, doc_name)
                    for doc_name in filenames
                    if doc_name.split('.')[-1].lower() in Global_stuff.ACCEPTED_EXTENSIONS
                ])
                if Global_stuff.DEPTH == 0:
                    break
            self.documents = [Document(p) for p in documents_paths]
        else:
            self.documents = []

        self.config = config or ComparisonConfig()
        self.corpus_weights = None
        self.similarities = []
        self.keyword_matches = {}

    def add_doc(self, document: Document):
        if self.index(document) == -1:
            self.documents.append(document)
            self.corpus_weights = None
            return f"Document {document.name} ajouté"
        return f"Document \"{document.name}\" déjà présent"

    def get_documents_names(self):
        return [d.name for d in self.documents]

    def index(self, document):
        names = self.get_documents_names()
        return names.index(document.name) if document.name in names else -1

    def __len__(self):
        return len(self.documents)

    def __add__(self, corpus_2):
        corpus_sum = Corpus(config=self.config)
        corpus_sum.documents = self.documents + [d for d in corpus_2.documents if d.name not in self.get_documents_names()]
        return corpus_sum

    def __sub__(self, document, inplace=False):
        corpus_sub = self if inplace else self.copy()
        index = self.index(document) if isinstance(document, Document) else document
        if index != -1:
            corpus_sub.documents.pop(index)
            corpus_sub.corpus_weights = None
        return corpus_sub

    def copy(self):
        corpus_copy = Corpus(config=self.config)
        corpus_copy.documents = copy(self.documents)
        return corpus_copy

    def filter(self, query: str, ignore_case: bool = True):
        """Filtre par mots-clés. Retourne un nouveau Corpus filtré."""
        filtered_corpus = Corpus(config=self.config)

        if ignore_case:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            scores = [len(pattern.findall(d.text.origin_content)) for d in self.documents]
        else:
            scores = [d.text.origin_content.count(query) for d in self.documents]

        filtered_indices = [i for i in range(len(self)) if scores[i] > 0]
        filtered_indices.sort(key=lambda i: scores[i], reverse=True)

        filtered_corpus.documents = [self.documents[i] for i in filtered_indices]
        filtered_corpus.keyword_matches = {self.documents[i].name: scores[i] for i in filtered_indices}

        return filtered_corpus

    def compute_corpus_weights(self, model):
        """Calcule TF-IDF ou BM25 sur l'ENSEMBLE du corpus."""
        corpus_texts = [d.text.origin_content for d in self.documents]

        all_weights = []
        for doc in self.documents:
            sentences = [s.content for s in doc.text.sentences]
            if len(sentences) == 0:
                all_weights.append(np.array([1.0]))
                continue

            if self.config.scoring_method == "bm25":
                weights = compute_bm25_weights(
                    corpus_texts, sentences,
                    k1=self.config.bm25_k1, b=self.config.bm25_b
                )
            else:
                weights = compute_tfidf_weights(corpus_texts, sentences)

            all_weights.append(weights)

        self.corpus_weights = all_weights
        return all_weights

    def vectorize_corpus(self, model):
        """Vectorise avec poids pré-calculés et normalisation L2."""
        if self.corpus_weights is None:
            self.compute_corpus_weights(model)

        corpus_matrix = []
        for i, doc in enumerate(self.documents):
            vec = doc.vectorize_document(
                model, self.corpus_weights[i],
                config=self.config,
                normalize=self.config.normalize_embeddings
            )
            corpus_matrix.append(vec)

        corpus_matrix = np.array(corpus_matrix)

        if self.config.normalize_embeddings:
            corpus_matrix = normalize_embeddings(corpus_matrix, method="l2")

        return corpus_matrix

    def compare(self, source, model=None, n=0, inplace=True):
        """Compare par similarité avec le document source."""
        if model is None:
            model = SentenceTransformer(self.config.model_name)

        if n <= 0 or n > len(self):
            n = len(self)

        corpus = self.copy()

        if isinstance(source, int):
            index = source
        elif isinstance(source, Document):
            index = corpus.index(source)
            if index == -1:
                corpus.add_doc(source)
                index = len(corpus) - 1

        corpus.compute_corpus_weights(model)
        corpus_matrix = corpus.vectorize_corpus(model)

        source_vector = corpus_matrix[index].reshape(1, -1)
        cosine_scores = cosine_similarity(source_vector, corpus_matrix)[0]

        corpus_texts = [d.text.origin_content for d in corpus.documents]
        source_text  = corpus_texts[index]
        query_words  = source_text.lower().split()

        bm25_index = BM25(corpus_texts, k1=self.config.bm25_k1, b=self.config.bm25_b)

        avgdl = bm25_index.avgdl
        k1, b = bm25_index.k1, bm25_index.b
        bm25_per_doc = np.zeros(len(corpus.documents))
        for doc_idx, doc_freq in enumerate(bm25_index.doc_freqs):
            dl = sum(doc_freq.values())
            score = 0.0
            for word in query_words:
                word_id = bm25_index.vocab.get(word, -1)
                if word_id < 0:
                    continue
                tf = doc_freq.get(word_id, 0)
                idf = bm25_index.idf.get(word_id, 0.0)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
            bm25_per_doc[doc_idx] = score

        bm25_min, bm25_max = bm25_per_doc.min(), bm25_per_doc.max()
        if bm25_max > bm25_min:
            bm25_norm = (bm25_per_doc - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = np.ones_like(bm25_per_doc)

        alpha = self.config.semantic_weight
        similarities = alpha * cosine_scores + (1.0 - alpha) * bm25_norm

        best_indices = np.argsort(similarities)[::-1]

        sorted_corpus = Corpus(config=self.config)
        sorted_corpus.documents = [corpus.documents[i] for i in best_indices[:n]]
        sorted_corpus.similarities = [similarities[i] for i in best_indices[:n]]

        if inplace:
            self.documents = sorted_corpus.documents
            self.similarities = sorted_corpus.similarities

        return sorted_corpus
