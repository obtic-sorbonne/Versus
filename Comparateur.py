"""
Comparateur : alignement fin Document–Document.

Architecture en entonnoir :
  1. Segmentation par défaut en phrases
  2. Alignement par similarité cosinus (ANN/FAISS ou exact)
  3. Détection des correspondances, suppressions et insertions
  4. Affinage optionnel par n-grams glissants sur zones intermédiaires
  5. Agrégation bidirectionnelle optionnelle

Visualisation par segment aligné :
  - Correspondances (vert) : texte commun aux deux segments
  - Suppressions (rouge)   : texte présent dans la source, absent de la cible
  - Insertions (bleu)      : texte présent dans la cible, absent de la source
"""

import numpy as np
from Global_stuff import Global_stuff
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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
    #  Méthode principale — point d'entrée
    # =================================================================

    def compare_n_grams(self, n=3, model=None, score_threshold=None):
        """
        Comparaison complète en entonnoir.
        Retourne une liste de tuples (pos1, pos2, score) sans diff.
        Utiliser compute_diffs() pour calculer les suppressions/insertions
        sur un sous-ensemble de résultats.
        """
        if self.config.debug:
            import time as _time
            _profile = {}
            _t0 = _time.time()

        if model is None:
            model = SentenceTransformer(self.config.model_name)

        if score_threshold is None:
            score_threshold = self.config.similarity_threshold

        # ── Passe 1 : alignement par phrases ──
        if self.config.debug:
            _t = _time.time()
        raw_alignments = self._align_sentences(model, score_threshold)
        if self.config.debug:
            _profile['Passe 1 (phrases)'] = _time.time() - _t

        # ── Passe 2 : affinage n-grams sur zones intermédiaires ──
        if self.config.ngram_refinement:
            if self.config.debug:
                _t = _time.time()
            extra = self._refine_intermediate_zones(
                raw_alignments, model, n, score_threshold
            )
            raw_alignments = self._merge_alignments(raw_alignments, extra)
            if self.config.debug:
                _profile['Passe 2 (n-grams)'] = _time.time() - _t

        # ── Agrégation bidirectionnelle ──
        if self.config.bidirectional:
            if self.config.debug:
                _t = _time.time()
            reverse_alignments = self._align_sentences(model, score_threshold, reverse=True)
            if self.config.debug:
                _profile['Bidirectionnel (phrases)'] = _time.time() - _t
            if self.config.ngram_refinement:
                if self.config.debug:
                    _t = _time.time()
                extra_rev = self._refine_intermediate_zones(
                    reverse_alignments, model, n, score_threshold
                )
                reverse_alignments = self._merge_alignments(reverse_alignments, extra_rev)
                if self.config.debug:
                    _profile['Bidirectionnel (n-grams)'] = _time.time() - _t
            raw_alignments = self._merge_alignments(raw_alignments, reverse_alignments)

        # ── Fusion sémantique / lexicale (70/30 par défaut) ──
        if self.config.debug:
            _t = _time.time()
        result = self._apply_combined_score(raw_alignments, score_threshold)

        if self.config.debug:
            _profile['Préparation résultats'] = _time.time() - _t
            _profile['TOTAL'] = _time.time() - _t0
            print("\n" + "=" * 50)
            print("  PROFILING compare_n_grams()")
            print("=" * 50)
            for step, elapsed in _profile.items():
                bar = "█" * int(elapsed / _profile['TOTAL'] * 30) if _profile['TOTAL'] > 0 else ""
                print(f"  {step:30s} {elapsed:7.2f}s  {bar}")
            print("=" * 50 + "\n")
            self._last_profile = _profile

        return result

    def _apply_combined_score(self, alignments, score_threshold):
        """
        Fusionne le score cosinus (embeddings), le score Jaccard et le score Overlap (lexicaux)
        selon Option B : 60% sémantique + 20% Jaccard + 20% Overlap (normalisation min-max).
        Stocke les scores bruts dans self._raw_scores pour permettre le re-tri ultérieur.
        Retourne [(pos1, pos2, score_combiné), ...].
        """
        import re as _re

        sw = Global_stuff.STOPWORDS

        def _words(t):
            return {w.lower() for w in _re.findall(r'\b\w+\b', t)
                    if len(w) > 2 and w.lower() not in sw}

        def jaccard(pos1, pos2):
            w1 = _words(self.text1.origin_content[pos1[0]:pos1[1]])
            w2 = _words(self.text2.origin_content[pos2[0]:pos2[1]])
            if not w1 and not w2:
                return 0.0
            return len(w1 & w2) / len(w1 | w2)

        def overlap(pos1, pos2):
            w1 = _words(self.text1.origin_content[pos1[0]:pos1[1]])
            w2 = _words(self.text2.origin_content[pos2[0]:pos2[1]])
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / min(len(w1), len(w2))

        if not alignments:
            self._raw_scores = {}
            return []

        # Calcul des scores bruts
        cosine_scores = [score for (_, _, score) in alignments]
        jac_scores    = [jaccard(pos1, pos2) for (pos1, pos2, _) in alignments]
        ovl_scores    = [overlap(pos1, pos2) for (pos1, pos2, _) in alignments]

        # Normalisation min-max de chaque série
        def minmax(values):
            lo, hi = min(values), max(values)
            if hi == lo:
                return [1.0] * len(values)
            return [(v - lo) / (hi - lo) for v in values]

        cos_norm = minmax(cosine_scores)
        jac_norm = minmax(jac_scores)
        ovl_norm = minmax(ovl_scores)

        sw_ = self.config.semantic_weight   # 0.60
        lw_ = self.config.lexical_weight    # 0.40 → 20% Jaccard + 20% Overlap

        # Stockage des scores bruts indexés par position pour re-tri ultérieur
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

        # Tri par défaut : score combiné décroissant
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
        """
        result = []
        for item in alignments:
            pos1, pos2 = item[0], item[1]
            t1 = self.text1[pos1]
            t2 = self.text2[pos2]

            suppressions, insertions = [], []
            s = SequenceMatcher(lambda x: x == " ", t1, t2)
            for action, i1, i2, j1, j2 in s.get_opcodes():
                if action == 'delete' or action == 'replace':
                    suppressions.append((pos1[0] + i1, pos1[0] + i2))
                if action == 'insert' or action == 'replace':
                    insertions.append((pos2[0] + j1, pos2[0] + j2))

            result.append((pos1, pos2, suppressions, insertions))
        return result

    # =================================================================
    #  Passe 1 : Alignement par phrases
    # =================================================================

    def _align_sentences(self, model, score_threshold, reverse=False):
        """
        Encode les phrases, recherche ANN ou cosinus exact,
        applique seuil fixe ou adaptatif.
        Si reverse=True, aligne dans la direction inverse (target → source).
        Retourne [(pos1, pos2, score), ...].
        """
        if reverse:
            sentences1 = self.text2.sentences
            sentences2 = self.text1.sentences
        else:
            sentences1 = self.text1.sentences
            sentences2 = self.text2.sentences

        if not sentences1 or not sentences2:
            return []

        contents1 = [s.content for s in sentences1]
        contents2 = [s.content for s in sentences2]

        emb1 = model.encode(contents1, show_progress_bar=False, batch_size=256)
        emb2 = model.encode(contents2, show_progress_bar=False, batch_size=256)

        emb1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
        emb2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)

        if FAISS_AVAILABLE and len(sentences2) > 50:
            raw = self._search_ann(emb1, emb2, sentences1, sentences2, score_threshold)
        else:
            raw = self._search_exact(emb1, emb2, sentences1, sentences2, score_threshold)

        if reverse:
            return [(pos2, pos1, score) for (pos1, pos2, score) in raw]
        return raw

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
    #  Passe 2 : Affinage n-grams sur zones intermédiaires
    # =================================================================

    def _refine_intermediate_zones(self, sentence_alignments, model, n, score_threshold):
        low, high = self.config.ngram_zone_low, self.config.ngram_zone_high

        best_by_source = {}
        for (pos1, pos2, score) in sentence_alignments:
            if pos1 not in best_by_source or score > best_by_source[pos1]:
                best_by_source[pos1] = score

        sentences1 = self.text1.sentences
        candidate_indices_1 = set()

        for i, s in enumerate(sentences1):
            pos = (s.index, s.index + len(s.content))
            best = best_by_source.get(pos, 0.0)
            if low <= best <= high:
                candidate_indices_1.add(i)

        if not candidate_indices_1:
            return []

        n_grams1 = self.text1.n_grams(n)
        n_grams2 = self.text2.n_grams(n)

        if not n_grams1 or not n_grams2:
            return []

        w2s_map = self._word_to_sentence_map(self.text1)
        sent_ranges = self._sentence_to_ngram_ranges(w2s_map, len(self.text1.words), n)

        margin = n
        max_ng1 = len(n_grams1)
        active_set = set()
        for si in candidate_indices_1:
            if si in sent_ranges:
                lo, hi = sent_ranges[si]
                for idx in range(max(0, lo - margin), min(max_ng1, hi + margin)):
                    active_set.add(idx)

        if not active_set:
            return []

        active_idx1 = sorted(active_set)
        active_idx2 = list(range(len(n_grams2)))

        rows, cols, data = self._compare_tfidf(
            n_grams1, n_grams2, active_idx1, active_idx2, n, score_threshold
        )

        if not data:
            return []

        aggregated = self._aggregate_diagonal(rows, cols, data, n_grams1, n_grams2, n)
        return [(pos1, pos2, 0.0) for (pos1, pos2) in aggregated]

    # =================================================================
    #  TF-IDF sur n-grams
    # =================================================================

    def _compare_tfidf(self, n_grams1, n_grams2, active_idx1, active_idx2, n, score_threshold):
        content_1 = [" ".join(w.content.lower() for w in n_grams1[i]) for i in active_idx1]
        content_2 = [" ".join(w.content.lower() for w in n_grams2[j]) for j in active_idx2]

        if not content_1 or not content_2:
            return [], [], []

        vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\S+')
        vectorizer.fit(content_1 + content_2)

        tfidf_1 = vectorizer.transform(content_1)
        tfidf_2 = vectorizer.transform(content_2)

        all_rows, all_cols, all_data = [], [], []
        step = 5000

        for i in range(0, tfidf_1.shape[0], step):
            end_i = min(i + step, tfidf_1.shape[0])
            sim_block = cosine_similarity(tfidf_1[i:end_i], tfidf_2)

            if self.config.adaptive_threshold:
                for local_i in range(sim_block.shape[0]):
                    global_i = active_idx1[i + local_i]
                    seg_len = sum(len(w.content) for w in n_grams1[global_i])
                    th = self.config.get_adaptive_threshold(seg_len)
                    hits = np.where(sim_block[local_i] >= th)[0]
                    for j in hits:
                        all_rows.append(global_i)
                        all_cols.append(active_idx2[j])
                        all_data.append(float(sim_block[local_i, j]))
            else:
                local_rows, local_cols = np.where(sim_block >= score_threshold)
                for lr, lc in zip(local_rows, local_cols):
                    all_rows.append(active_idx1[i + lr])
                    all_cols.append(active_idx2[lc])
                    all_data.append(float(sim_block[lr, lc]))

        return all_rows, all_cols, all_data

    # =================================================================
    #  Mapping mots ↔ phrases
    # =================================================================

    def _word_to_sentence_map(self, text):
        sentences = text.sentences
        if not sentences:
            return np.zeros(len(text.words), dtype=int)
        boundaries = []
        for i, s in enumerate(sentences):
            start = s.index
            end = sentences[i + 1].index if i + 1 < len(sentences) else float('inf')
            boundaries.append((start, end))
        mapping = np.zeros(len(text.words), dtype=int)
        sent_idx = 0
        for w_idx, word in enumerate(text.words):
            while sent_idx < len(boundaries) - 1 and word.start >= boundaries[sent_idx][1]:
                sent_idx += 1
            mapping[w_idx] = sent_idx
        return mapping

    def _sentence_to_ngram_ranges(self, word_sent_map, n_words, n):
        n_ngrams = n_words - n + 1
        if n_ngrams <= 0:
            return {}
        ranges = {}
        current_sent = -1
        range_start = 0
        for ng_idx in range(n_ngrams):
            sent = int(word_sent_map[ng_idx])
            if sent != current_sent:
                if current_sent >= 0:
                    ranges[current_sent] = (range_start, ng_idx)
                current_sent = sent
                range_start = ng_idx
        if current_sent >= 0:
            ranges[current_sent] = (range_start, n_ngrams)
        return ranges

    # =================================================================
    #  Agrégation diagonale
    # =================================================================

    def _aggregate_diagonal(self, all_rows, all_cols, all_data, n_grams1, n_grams2, n):
        aggreg = {}
        to_remove = set()
        indices = sorted(range(len(all_rows)), key=lambda k: (all_rows[k], all_cols[k]))
        for idx in indices:
            row_ind, col_ind = all_rows[idx], all_cols[idx]
            g1, g2 = n_grams1[row_ind], n_grams2[col_ind]
            score = all_data[idx]
            prev_key = (row_ind - 1, col_ind - 1)
            if prev_key in aggreg and prev_key not in to_remove:
                prec = aggreg[prev_key]
                if n > 1:
                    new_gram1 = list(prec[0][:-n+1]) + list(g1)
                    new_gram2 = list(prec[1][:-n+1]) + list(g2)
                else:
                    new_gram1 = list(prec[0]) + list(g1)
                    new_gram2 = list(prec[1]) + list(g2)
                new_score = (prec[2] + score) / 2
                aggreg[(row_ind, col_ind)] = (new_gram1, new_gram2, new_score)
                to_remove.add(prev_key)
            else:
                aggreg[(row_ind, col_ind)] = (list(g1), list(g2), score)
        for key in to_remove:
            aggreg.pop(key, None)
        result = []
        for (row, col), (words1, words2, sc) in sorted(aggreg.items(), key=lambda x: x[0][0]):
            if words1 and words2:
                result.append((
                    (words1[0].start, words1[-1].end),
                    (words2[0].start, words2[-1].end)
                ))
        return result

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

    def _merge_alignments(self, a1, a2):
        best = {}
        for (pos1, pos2, score) in a1 + a2:
            key = (pos1, pos2)
            if key not in best or score > best[key][2]:
                best[key] = (pos1, pos2, score)
        return sorted(best.values(), key=lambda x: x[2], reverse=True)

    def remove_stopwords_texts(self):
        for t in (self.text1, self.text2):
            t.remove_stopwords()

    def set_default_texts(self):
        for t in (self.text1, self.text2):
            t.default()
