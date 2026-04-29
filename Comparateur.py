"""
Comparateur : alignement fin Document–Document.

Architecture :
  1. Segmentation par phrases
  2. Alignement par similarité cosinus (ANN/FAISS ou exact)
  3. Fusion sémantique / lexicale (score combiné)
"""

import numpy as np
from Global_stuff import Global_stuff
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Config import ComparisonConfig

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class PairText:
    def __init__(self, text1, text2, config: ComparisonConfig = None):
        self.text1 = text1
        self.text2 = text2
        self.config = config or ComparisonConfig()

    # =================================================================
    #  Méthode principale
    # =================================================================

    def compare_n_grams(self, n=3, model=None, score_threshold=None):
        """
        Comparaison complète.
        Retourne une liste de tuples (pos1, pos2, score).
        """
        if model is None:
            model = SentenceTransformer(self.config.model_name)

        if score_threshold is None:
            score_threshold = self.config.similarity_threshold

        raw_alignments = self._align_sentences(model, score_threshold)
        return self._apply_combined_score(raw_alignments)

    def _apply_combined_score(self, alignments):
        """
        Fusionne cosinus (sémantique), Jaccard et Overlap (lexicaux).
        60% sémantique + 20% Jaccard + 20% Overlap (normalisation min-max).
        Stocke les scores bruts dans self._raw_scores pour le re-tri.
        Optimisation : w1/w2 calculés une seule fois par alignement.
        """
        import re as _re

        sw = Global_stuff.STOPWORDS

        def _words(t):
            return {w.lower() for w in _re.findall(r'\b\w+\b', t)
                    if len(w) > 2 and w.lower() not in sw}

        if not alignments:
            self._raw_scores = {}
            return []

        cosine_scores = []
        jac_scores    = []
        ovl_scores    = []

        # Un seul calcul de w1/w2 par alignement pour Jaccard ET Overlap
        for (pos1, pos2, score) in alignments:
            cosine_scores.append(score)
            w1 = _words(self.text1.origin_content[pos1[0]:pos1[1]])
            w2 = _words(self.text2.origin_content[pos2[0]:pos2[1]])
            inter = len(w1 & w2)
            union = len(w1 | w2)
            jac_scores.append(inter / union if union else 0.0)
            ovl_scores.append(inter / min(len(w1), len(w2)) if w1 and w2 else 0.0)

        def minmax(values):
            lo, hi = min(values), max(values)
            if hi == lo:
                return [1.0] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        cos_norm = minmax(cosine_scores)
        jac_norm = minmax(jac_scores)
        ovl_norm = minmax(ovl_scores)

        sw_ = self.config.semantic_weight
        lw_ = self.config.lexical_weight

        self._raw_scores = {}
        result = []
        for idx, (pos1, pos2, _) in enumerate(alignments):
            combined = sw_ * cos_norm[idx] + (lw_ / 2) * jac_norm[idx] + (lw_ / 2) * ovl_norm[idx]
            self._raw_scores[(pos1, pos2)] = {
                'semantic':        cosine_scores[idx],
                'lexical_jaccard': jac_scores[idx],
                'lexical_overlap': ovl_scores[idx],
                'combined':        combined,
            }
            result.append((pos1, pos2, combined))

        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def sort_alignments(self, alignments, mode="combined"):
        """
        Re-trie une liste d'alignements selon le mode choisi.
        mode : "combined" | "semantic" | "lexical_jaccard" | "lexical_overlap"
        """
        raw = getattr(self, '_raw_scores', {})
        key_map = {
            "combined":        "combined",
            "semantic":        "semantic",
            "lexical_jaccard": "lexical_jaccard",
            "lexical_overlap": "lexical_overlap",
        }
        key = key_map.get(mode, "combined")
        return sorted(alignments, key=lambda x: raw.get((x[0], x[1]), {}).get(key, 0.0), reverse=True)

    def compute_diffs(self, alignments):
        """
        Calcule les suppressions/insertions pour un sous-ensemble d'alignements.
        Entrée:  [(pos1, pos2, score), ...]
        Sortie:  [(pos1, pos2, suppressions, insertions), ...]

        Optimisations :
        - Diff au niveau MOT (au lieu de caractère) → beaucoup plus rapide
        - autojunk=False pour éviter le surcoût de détection automatique
        - Parallélisation via ThreadPoolExecutor (4 workers)
        """
        import re
        from concurrent.futures import ThreadPoolExecutor

        def _tokenize(text, base_offset):
            """Retourne liste de (mot, start_abs, end_abs)."""
            return [(m.group(), base_offset + m.start(), base_offset + m.end())
                    for m in re.finditer(r'\S+', text)]

        def _diff_one(item):
            pos1, pos2 = item[0], item[1]
            t1 = self.text1[pos1]
            t2 = self.text2[pos2]

            tokens1 = _tokenize(t1, pos1[0])
            tokens2 = _tokenize(t2, pos2[0])

            words1 = [w for w, _, _ in tokens1]
            words2 = [w for w, _, _ in tokens2]

            suppressions, insertions = [], []
            sm = SequenceMatcher(None, words1, words2, autojunk=False)
            for action, i1, i2, j1, j2 in sm.get_opcodes():
                if action in ('delete', 'replace') and i1 < i2:
                    suppressions.append((tokens1[i1][1], tokens1[i2 - 1][2]))
                if action in ('insert', 'replace') and j1 < j2:
                    insertions.append((tokens2[j1][1], tokens2[j2 - 1][2]))

            return (pos1, pos2, suppressions, insertions)

        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(_diff_one, alignments))

    # =================================================================
    #  Alignement par phrases
    # =================================================================

    def _align_sentences(self, model, score_threshold):
        """
        Encode les phrases, recherche ANN ou cosinus exact.
        Retourne [(pos1, pos2, score), ...].
        Réutilise sentence.vectorized si déjà calculé par vectorize_document(),
        évitant ainsi un double appel coûteux à model.encode().
        """
        sentences1 = self.text1.sentences
        sentences2 = self.text2.sentences

        if not sentences1 or not sentences2:
            return []

        if all(s.vectorized is not None for s in sentences1):
            emb1 = np.array([s.vectorized for s in sentences1])
        else:
            emb1 = model.encode([s.content for s in sentences1],
                                 show_progress_bar=False, batch_size=256)

        if all(s.vectorized is not None for s in sentences2):
            emb2 = np.array([s.vectorized for s in sentences2])
        else:
            emb2 = model.encode([s.content for s in sentences2],
                                 show_progress_bar=False, batch_size=256)

        emb1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
        emb2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)

        if FAISS_AVAILABLE and len(sentences2) > 50:
            return self._search_ann(emb1, emb2, sentences1, sentences2, score_threshold)
        else:
            return self._search_exact(emb1, emb2, sentences1, sentences2, score_threshold)

    def _search_ann(self, emb1, emb2, sentences1, sentences2, score_threshold):
        """Recherche ANN via FAISS."""
        dim = emb2.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb2.astype(np.float32))

        k = min(self.config.ann_k, len(sentences2))
        scores, indices = index.search(emb1.astype(np.float32), k)

        alignments = []
        for i in range(len(sentences1)):
            s1 = sentences1[i]
            th = self._get_threshold(s1.content)

            for j_idx in range(k):
                score = float(scores[i, j_idx])
                j = int(indices[i, j_idx])
                if j < 0:
                    continue
                if score >= th:
                    s2 = sentences2[j]
                    pos1 = (s1.index, getattr(s1, '_end', s1.index + len(s1.content)))
                    pos2 = (s2.index, getattr(s2, '_end', s2.index + len(s2.content)))
                    alignments.append((pos1, pos2, score))

        return self._deduplicate(alignments)

    def _search_exact(self, emb1, emb2, sentences1, sentences2, score_threshold):
        """Recherche exacte par blocs (chunking)."""
        alignments = []
        chunk_size = self.config.chunk_size

        for start in range(0, len(sentences1), chunk_size):
            end = min(start + chunk_size, len(sentences1))
            sim_block = cosine_similarity(emb1[start:end], emb2)

            for i_local in range(sim_block.shape[0]):
                i_global = start + i_local
                s1 = sentences1[i_global]
                th = self._get_threshold(s1.content)

                hits = np.where(sim_block[i_local] >= th)[0]
                for j in hits:
                    score = float(sim_block[i_local, j])
                    s2 = sentences2[j]
                    pos1 = (s1.index, getattr(s1, '_end', s1.index + len(s1.content)))
                    pos2 = (s2.index, getattr(s2, '_end', s2.index + len(s2.content)))
                    alignments.append((pos1, pos2, score))

        return self._deduplicate(alignments)

    # =================================================================
    #  Utilitaires
    # =================================================================

    def _get_threshold(self, text):
        if self.config.adaptive_threshold:
            seg_len = len(text.split())
            return self.config.get_adaptive_threshold(seg_len)
        return self.config.similarity_threshold

    def _deduplicate(self, alignments):
        best = {}
        for (pos1, pos2, score) in alignments:
            key = (pos1, pos2)
            if key not in best or score > best[key][2]:
                best[key] = (pos1, pos2, score)
        return sorted(best.values(), key=lambda x: x[2], reverse=True)
