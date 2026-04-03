"""
Document : vectorisation avec normalisation L2 et pondération.

CORRECTIONS CRITIQUES APPLIQUÉES:
1. Normalisation L2 systématique des embeddings
2. Pondération selon stratégie configurable
3. Poids pré-calculés sur le corpus entier
"""

import numpy as np
from Text import Text
import chardet
from utils import normalize_embeddings, weight_embeddings
from Config import get_document_hash


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
            
            if self.vector_sentences is None:
                self.vector_sentences = model.encode(sentences_content)
                
                # CRITIQUE: Normalisation L2
                if normalize:
                    self.vector_sentences = normalize_embeddings(self.vector_sentences, method="l2")
            
            for i in range(t.n_sentences):
                if t.sentences[i].vectorized is None:
                    t.sentences[i].vectorized = self.vector_sentences[i]
            
            # Pondération selon stratégie
            strategy = config.weighting_strategy if config else "sqrt"
            self.vectorized = weight_embeddings(self.vector_sentences, weights, strategy=strategy)
            
            # Normalisation finale
            if normalize:
                norm = np.linalg.norm(self.vectorized)
                if norm > 0:
                    self.vectorized = self.vectorized / norm

        return self.vectorized
    
    def reset_vectorization(self):
        """Réinitialise pour recalcul."""
        self.vectorized = None
        self.vector_sentences = None
        for s in self.text.sentences:
            s.vectorized = None
    
    def get_info(self) -> dict:
        """Infos pour export."""
        return {
            "name": self.name,
            "hash": self.document_hash,
            "num_words": len(self.text.words),
            "num_sentences": self.text.n_sentences
        }
