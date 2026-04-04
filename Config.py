"""
Configuration centralisée pour VERSUS.
"""

import json
import hashlib
import math
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ComparisonConfig:
    """Configuration pour une comparaison de textes."""

    # Modèle
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Méthode de scoring (utilisée par le classement corpus)
    scoring_method: str = "bm25"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    normalize_embeddings: bool = True
    weighting_strategy: str = "sqrt"

    # Seuils
    similarity_threshold: float = 0.85
    adaptive_threshold: bool = True
    adaptive_alpha: float = 0.05

    # ANN (FAISS)
    ann_enabled: bool = True
    ann_k: int = 10

    # Chunking (fallback sans FAISS)
    chunk_size: int = 3000

    # Fusion sémantique / lexicale
    semantic_weight: float = 0.60
    lexical_weight: float = 0.40

    # Stopwords
    use_stopwords: bool = False

    # Métadonnées
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def get_adaptive_threshold(self, segment_length: int) -> float:
        """Calcule le seuil adaptatif selon la longueur du segment."""
        if not self.adaptive_threshold:
            return self.similarity_threshold
        adjustment = self.adaptive_alpha * math.log(max(segment_length, 1))
        return max(0.5, self.similarity_threshold - adjustment)

    def to_dict(self):
        return asdict(self)

    def to_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict):
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def get_hash(self) -> str:
        """Hash unique de la configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_document_hash(content: str) -> str:
    """Hash MD5 d'un document."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
