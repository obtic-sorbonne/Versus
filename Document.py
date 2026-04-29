"""
Document : vectorisation avec normalisation L2 et pondération.
Cache mémoire des embeddings via st.cache_resource (Streamlit Community Cloud compatible).
"""

import numpy as np
from Text import Text
import chardet
from utils import normalize_embeddings, weight_embeddings
from Config import get_document_hash


def _get_app_cache():
    """Retourne le dictionnaire de cache app-level géré par st.cache_resource."""
    try:
        from App_st import get_embed_cache
        return get_embed_cache()
    except Exception:
        return {}


class Document:
    """Document : vectorisation et comparaison."""

    def __init__(self, name, file_value=None, content=None):
        self.name = name
        self.vectorized = None
        self.vector_sentences = None
        self.document_hash = None

        if file_value:
            encoding = chardet.detect(file_value)['encoding'] or 'utf-8'
            content_str = file_value.decode(encoding, errors='replace')
            self.text = Text(self.name, content_str)
            self.document_hash = get_document_hash(content_str)
        elif content:
            self.text = Text(self.name, content)
            self.document_hash = get_document_hash(content)

    def vectorize_document(self, model, weights, config=None, normalize=True):
        """
        Vectorise avec normalisation L2 et pondération.

        Args:
            model: SentenceTransformer
            weights: Poids pré-calculés sur le corpus entier (CRITIQUE)
            config: ComparisonConfig
            normalize: Appliquer normalisation L2
        """
        if self.vectorized is None:
            t = self.text
            sentences_content = [s.content for s in t.sentences]
            n = len(sentences_content)

            if self.vector_sentences is None:
                # 1. Chercher dans le cache mémoire app-level
                cache = _get_app_cache()
                cached = cache.get(self.document_hash) if self.document_hash else None

                if cached is not None and cached.shape[0] == n:
                    self.vector_sentences = cached
                else:
                    # 2. Calculer les embeddings (batch_size=64 pour le throughput)
                    self.vector_sentences = model.encode(
                        sentences_content,
                        batch_size=64,
                        show_progress_bar=False,
                    )
                    if normalize:
                        self.vector_sentences = normalize_embeddings(
                            self.vector_sentences, method="l2"
                        )
                    # 3. Stocker dans le cache app-level
                    if self.document_hash and cache is not None:
                        cache[self.document_hash] = self.vector_sentences

            for i in range(t.n_sentences):
                if t.sentences[i].vectorized is None:
                    t.sentences[i].vectorized = self.vector_sentences[i]

            strategy = config.weighting_strategy if config else "sqrt"
            self.vectorized = weight_embeddings(self.vector_sentences, weights, strategy=strategy)

            if normalize:
                norm = np.linalg.norm(self.vectorized)
                if norm > 0:
                    self.vectorized = self.vectorized / norm

        return self.vectorized
