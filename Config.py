"""
Configuration centralisée et profils prédéfinis pour VERSUS.

Mise à jour : segmentation par phrases par défaut,
affinage n-grams ciblé sur zones intermédiaires,
ANN (FAISS), agrégation bidirectionnelle.
"""

import json
import hashlib
import math
from dataclasses import dataclass, asdict
from typing import Literal, Tuple
from datetime import datetime


@dataclass
class ComparisonConfig:
    """Configuration complète pour une comparaison de textes."""
    
    # Modèle
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Méthode de scoring
    scoring_method: Literal["tfidf", "bm25"] = "bm25"
    
    # Pondération des embeddings
    weighting_strategy: Literal["mean", "sqrt", "weighted_sum"] = "sqrt"
    
    # Normalisation
    normalize_embeddings: bool = True
    
    # Seuils
    similarity_threshold: float = 0.85
    adaptive_threshold: bool = True
    adaptive_alpha: float = 0.05
    
    # N-grams : affinage ciblé sur zones intermédiaires
    ngram_refinement: bool = False
    ngram_size: int = 3
    ngram_zone_low: float = 0.60
    ngram_zone_high: float = 0.85
    
    # ANN (FAISS)
    ann_enabled: bool = True
    ann_k: int = 10
    
    # Agrégation bidirectionnelle
    bidirectional: bool = False
    
    # Chunking
    chunk_size: int = 3000
    
    # Fusion sémantique / lexicale
    semantic_weight: float = 0.60   # poids du score embeddings (cosinus)
    lexical_weight: float = 0.40    # poids du score Jaccard (lexical)

    # Stopwords
    use_stopwords: bool = False
    
    # Paramètres BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Fusion segments
    merge_gap: int = 5
    
    # Métadonnées
    profile_name: str = "custom"
    created_at: str = ""

    # Debug
    debug: bool = False
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    @property
    def ngram_zone(self) -> Tuple[float, float]:
        return (self.ngram_zone_low, self.ngram_zone_high)
    
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


class ConfigProfiles:
    """Profils prédéfinis — cahier des charges final."""
    
    @staticmethod
    def exploratory() -> ComparisonConfig:
        """Rappel élevé, trouve plus de correspondances."""
        return ComparisonConfig(
            profile_name="exploratory",
            scoring_method="bm25",
            weighting_strategy="sqrt",
            similarity_threshold=0.75,
            adaptive_threshold=True,
            adaptive_alpha=0.08,
            ngram_refinement=True,
            ngram_size=3,
            ngram_zone_low=0.60,
            ngram_zone_high=0.85,
            ann_enabled=True,
            bidirectional=False,
            use_stopwords=True
        )
    
    @staticmethod
    def philological() -> ComparisonConfig:
        """Précision maximale, analyse rigoureuse."""
        return ComparisonConfig(
            profile_name="philological",
            scoring_method="tfidf",
            weighting_strategy="mean",
            similarity_threshold=0.93,
            adaptive_threshold=False,
            ngram_refinement=True,
            ngram_size=5,
            ngram_zone_low=0.60,
            ngram_zone_high=0.93,
            ann_enabled=True,
            bidirectional=True,
            use_stopwords=False
        )
    
    @staticmethod
    def comparative() -> ComparisonConfig:
        """Équilibre entre précision et rappel."""
        return ComparisonConfig(
            profile_name="comparative",
            scoring_method="bm25",
            weighting_strategy="sqrt",
            similarity_threshold=0.85,
            adaptive_threshold=True,
            adaptive_alpha=0.05,
            ngram_refinement=True,
            ngram_size=4,
            ngram_zone_low=0.60,
            ngram_zone_high=0.85,
            ann_enabled=True,
            bidirectional=False,
            use_stopwords=True
        )
    
    @staticmethod
    def get_all_profiles():
        return {
            "philological": ConfigProfiles.philological(),
            "exploratory": ConfigProfiles.exploratory(),
            "comparative": ConfigProfiles.comparative()
        }
    
    @staticmethod
    def get_profile_descriptions():
        return {
            "philological": "📚 Alignement strict — Seuil=0.93, TF-IDF, n-grams (n=5), bidirectionnel",
            "exploratory": "🔍 Détection extensive — Seuil=0.75, BM25, affinage n-grams (n=3)",
            "comparative": "⚖️ Équilibre précision/rappel — Seuil=0.85, BM25, n-grams (n=4), adaptatif"
        }


def get_document_hash(content: str) -> str:
    """Hash MD5 d'un document."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
