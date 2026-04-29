"""
Guide d'utilisation — VERSUS
Contenu de la page Guide (onglet 5).
Appelé depuis App_st.py via render_guide().
"""

import streamlit as st
import streamlit.components.v1 as components


def render_guide():
    """Affiche le guide d'utilisation complet."""
    st.markdown("""
    <style>
        .guide-section {
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .guide-section h3 { color: #2563eb !important; font-size: 1rem; margin-bottom: 0.5rem; }
        .guide-section p, .guide-section li { font-size: 0.9rem; line-height: 1.6; color: #374151 !important; }
        .guide-badge {
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; color: #fff !important;
            margin-right: 4px; vertical-align: middle;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📖 Guide d'utilisation")
    st.caption("VERSUS — Outil de comparaison textuelle par similarité sémantique et lexicale")

    st.markdown("""
    <div class="guide-section">
        <h3>Principe général</h3>
        <ol>
            <li>On découpe le document source et le corpus cible en phrases.</li>
            <li>Chaque phrase est encodée en vecteur, puis pondérée par TF-IDF/BM25 selon son importance dans l'ensemble des documents.</li>
            <li>On fait une moyenne pondérée de ces vecteurs pour produire un seul vecteur représentatif par document.</li>
            <li>On classe les documents cibles par similarité décroissante en combinant similarité cosinus (sens) et BM25 (vocabulaire) dans un ratio 60/40.</li>
            <li>On prend les n documents les plus similaires et on lance la comparaison fine phrase à phrase.</li>
            <li>On récupère les embeddings de phrases des documents retenus (déjà calculés en étape&nbsp;2), puis on compare via similarité cosinus et un score combiné (60&nbsp;% cosinus + 20&nbsp;% Jaccard + 20&nbsp;% Overlap) pour retenir les paires les plus similaires.</li>
            <li>On affiche les paires alignées avec leur score, les différences surlignées (suppressions/insertions) et les mots en commun.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Diagramme pipeline ──
    pipeline_html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <style>
      body { margin: 0; padding: 0; background: transparent; font-family: system-ui, sans-serif; }
      svg text { font-family: system-ui, sans-serif; }
    </style>
    </head><body>
    <svg viewBox="0 0 900 760" width="100%" height="1000" xmlns="http://www.w3.org/2000/svg">

      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8"/>
        </marker>
        <marker id="arb" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#0284c7"/>
        </marker>
      </defs>

      <!-- Background -->
      <rect width="900" height="760" fill="#f8fafc" rx="12"/>

      <!-- Title -->
      <text x="20" y="30" text-anchor="start" font-size="14" font-weight="700" fill="#374151">Pipeline VERSUS — flux de traitement</text>

      <!-- ══════ COLONNE GAUCHE : PIPELINE ══════ -->

      <!-- ROW 1 : CHARGEMENT -->
      <rect x="30" y="50" width="155" height="68" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
      <text x="107" y="74" text-anchor="middle" font-size="11" font-weight="700" fill="#1d4ed8">📄 Texte source</text>
      <text x="107" y="91" text-anchor="middle" font-size="10" fill="#3b82f6">.txt / .docx / .pdf</text>
      <text x="107" y="107" text-anchor="middle" font-size="10" fill="#6b7280">segmentation en phrases</text>

      <rect x="205" y="50" width="155" height="68" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
      <text x="282" y="74" text-anchor="middle" font-size="11" font-weight="700" fill="#1d4ed8">📚 Corpus cible</text>
      <text x="282" y="91" text-anchor="middle" font-size="10" fill="#3b82f6">N documents</text>
      <text x="282" y="107" text-anchor="middle" font-size="10" fill="#6b7280">segmentation en phrases</text>

      <line x1="107" y1="118" x2="107" y2="152" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
      <line x1="282" y1="118" x2="282" y2="152" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>

      <!-- ROW 2 : FILTRE + PONDÉRATION -->
      <rect x="30" y="152" width="155" height="56" rx="8" fill="#f0fdf4" stroke="#10b981" stroke-width="1.5"/>
      <text x="107" y="175" text-anchor="middle" font-size="11" font-weight="700" fill="#059669">🔎 Filtre mots-clés</text>
      <text x="107" y="192" text-anchor="middle" font-size="10" fill="#6b7280">optionnel — regex</text>
      <text x="107" y="204" text-anchor="middle" font-size="9" fill="#10b981">corpus filtré → classement</text>

      <rect x="205" y="152" width="155" height="56" rx="8" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>
      <text x="282" y="175" text-anchor="middle" font-size="11" font-weight="700" fill="#92400e">⚙️ Pondération</text>
      <text x="282" y="192" text-anchor="middle" font-size="10" fill="#6b7280">BM25 / TF-IDF</text>
      <text x="282" y="204" text-anchor="middle" font-size="9" fill="#ca8a04">poids par phrase</text>

      <line x1="107" y1="208" x2="107" y2="242" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
      <line x1="282" y1="208" x2="282" y2="242" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>

      <!-- ROW 3 : CLASSEMENT -->
      <rect x="30" y="242" width="330" height="62" rx="8" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
      <text x="195" y="266" text-anchor="middle" font-size="11" font-weight="700" fill="#5b21b6">🏆 Classement par similarité globale</text>
      <text x="195" y="283" text-anchor="middle" font-size="10" fill="#6b7280">Embeddings · cosinus · tri décroissant</text>
      <text x="195" y="298" text-anchor="middle" font-size="9.5" fill="#7c3aed">→ sélection du document cible pour alignement fin</text>

      <line x1="195" y1="304" x2="195" y2="338" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>

      <!-- ROW 4 : ALIGNEMENT FIN -->
      <rect x="30" y="338" width="155" height="62" rx="8" fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
      <text x="107" y="360" text-anchor="middle" font-size="11" font-weight="700" fill="#9d174d">Passe 1</text>
      <text x="107" y="376" text-anchor="middle" font-size="10" font-weight="600" fill="#9d174d">Embeddings</text>
      <text x="107" y="392" text-anchor="middle" font-size="10" fill="#6b7280">cosinus · phrase × phrase</text>

      <rect x="205" y="338" width="155" height="62" rx="8" fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
      <text x="282" y="360" text-anchor="middle" font-size="11" font-weight="700" fill="#9d174d">Passe 2</text>
      <text x="282" y="376" text-anchor="middle" font-size="10" font-weight="600" fill="#9d174d">N-grams (optionnel)</text>
      <text x="282" y="392" text-anchor="middle" font-size="10" fill="#6b7280">TF-IDF · zone [0.60–0.85]</text>

      <line x1="107" y1="400" x2="107" y2="432" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
      <line x1="282" y1="400" x2="282" y2="432" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>

      <!-- ROW 5 : DIFF -->
      <rect x="30" y="432" width="330" height="56" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
      <text x="195" y="455" text-anchor="middle" font-size="11" font-weight="700" fill="#c2410c">Passe 3 — SequenceMatcher (diff)</text>
      <text x="195" y="471" text-anchor="middle" font-size="10" fill="#6b7280">suppressions · insertions · mots communs</text>
      <text x="195" y="483" text-anchor="middle" font-size="9" fill="#ea580c">affiché à la demande par bloc</text>

      <line x1="195" y1="488" x2="195" y2="518" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>

      <!-- ROW 6 : RÉSULTATS -->
      <rect x="30" y="518" width="155" height="34" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1.2"/>
      <text x="107" y="533" text-anchor="middle" font-size="10" font-weight="600" fill="#374151">📋 Blocs paginés</text>
      <text x="107" y="547" text-anchor="middle" font-size="9" fill="#6b7280">50 blocs / page</text>

      <rect x="205" y="518" width="155" height="34" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1.2"/>
      <text x="282" y="533" text-anchor="middle" font-size="10" font-weight="600" fill="#374151">↓ Export CSV</text>
      <text x="282" y="547" text-anchor="middle" font-size="9" fill="#6b7280">blocs sélectionnés</text>

      <!-- Légende (gauche, en bas) -->
      <rect x="30" y="578" width="330" height="162" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2"/>
      <text x="195" y="598" text-anchor="middle" font-size="11" font-weight="700" fill="#374151">Légende</text>

      <rect x="45" y="610" width="12" height="12" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/>
      <text x="62" y="621" font-size="10" fill="#374151">Entrée / chargement</text>
      <rect x="45" y="630" width="12" height="12" rx="2" fill="#f0fdf4" stroke="#10b981" stroke-width="1"/>
      <text x="62" y="641" font-size="10" fill="#374151">Filtre mots-clés (optionnel)</text>
      <rect x="45" y="650" width="12" height="12" rx="2" fill="#fef9c3" stroke="#ca8a04" stroke-width="1"/>
      <text x="62" y="661" font-size="10" fill="#374151">Pondération BM25 / TF-IDF</text>
      <rect x="45" y="670" width="12" height="12" rx="2" fill="#ede9fe" stroke="#7c3aed" stroke-width="1"/>
      <text x="62" y="681" font-size="10" fill="#374151">Classement global</text>
      <rect x="45" y="690" width="12" height="12" rx="2" fill="#fce7f3" stroke="#db2777" stroke-width="1"/>
      <text x="62" y="701" font-size="10" fill="#374151">Alignement fin (passes 1 &amp; 2)</text>
      <rect x="45" y="710" width="12" height="12" rx="2" fill="#fff7ed" stroke="#ea580c" stroke-width="1"/>
      <text x="62" y="721" font-size="10" fill="#374151">Diff — SequenceMatcher (passe 3)</text>

      <rect x="210" y="610" width="12" height="12" rx="2" fill="#f1f5f9" stroke="#64748b" stroke-width="1"/>
      <text x="227" y="621" font-size="10" fill="#374151">Résultats / Export</text>
      <line x1="210" y1="636" x2="228" y2="636" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>
      <text x="233" y="640" font-size="10" fill="#374151">flux principal</text>

    </svg>
    </body></html>"""
    components.html(pipeline_html, height=1050, scrolling=False)
    st.markdown("---")

    st.markdown("### Les 4 étapes")

    st.markdown("""
    <div class="guide-section">
        <h3>1 — Chargement</h3>
        <p>Chargez un <b>texte source</b> (à gauche) et un ou plusieurs <b>documents cibles</b> (à droite)
        au format <code>.txt</code>, <code>.docx</code> ou <code>.pdf</code>. Chaque fichier peut être
        supprimé via le bouton ✕. Cliquez sur <b>Valider →</b> pour passer au traitement.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-section">
        <h3>2 — Traitement</h3>
        <p>Choisissez un mode de classement :</p>
        <p><b>Par similarité</b> — vectorise et trie les documents cibles par score de similarité globale
        (combinaison cosinus + BM25). L'option <b>Exclure les stopwords</b> concentre la comparaison
        sur le vocabulaire significatif. Cliquez sur <b>Classer les textes par similarité →</b>.</p>
        <p><b>Par mots-clés</b> — filtre d'abord le corpus cible par un mot-clé (regex, insensible à
        la casse en option), puis classe les documents retenus par nombre d'occurrences décroissant.
        Cliquez sur <b>Filtrer</b> pour lancer le filtrage et le classement en une seule action.</p>
        <p>Sélectionnez ensuite un document cible dans le menu déroulant puis cliquez sur
        <b>Aligner les textes →</b> pour lancer l'alignement fin.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-section">
        <h3>3 — Résultats</h3>
        <p>Comparaison fine entre le texte source et le texte cible. Les résultats sont paginés par
        blocs de 50. Chaque bloc affiche trois scores :</p>
        <p>
            <b>sém.</b> (bleu) — score cosinus entre les embeddings. Mesure la proximité de sens,
            indépendamment de la forme lexicale.<br>
            <b>lex.</b> (vert/orange/rouge) — score Jaccard calculé sur les mots en commun hors stopwords.
            Vert ≥ 40 %, orange 15–39 %, rouge &lt; 15 %.<br>
            <b>cmb.</b> (violet) — fusion 60 % sémantique + 40 % lexical, après normalisation min-max.
        </p>
        <p>Le tri (<b>Combinée / Sémantique / Lexicale</b>) est accessible directement en haut des résultats.
        Les boutons <b>−</b> / <b>Reset</b> / <b>+</b> ajustent la fenêtre de contexte. Les cases à cocher
        permettent d'afficher :</p>
        <p>
            <span class="guide-badge" style="background:#dc2626">Suppressions</span> présentes dans la source, absentes de la cible.<br>
            <span class="guide-badge" style="background:#2563eb">Insertions</span> présentes dans la cible, absentes de la source.<br>
            <span class="guide-badge" style="background:#059669">Mots communs</span> termes partagés (hors stopwords) dans le contexte visible.<br><br>
            La case <b>Sélectionner</b> permet d'exporter via <b>↓ Export CSV</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-section">
        <h3>4 — Stats et Viz</h3>
        <p>Onglet dédié, disponible après alignement. Contient les statistiques comparatives
        (mots, phrases, diversité lexicale, fréquences communes top 10, mots propres top 10)
        et huit visualisations :</p>
        <p>
            <b>Carte des alignements (Heatmap)</b> — positionne chaque correspondance dans un repère source×cible ; la couleur indique le score (jaune → vert).<br>
            <b>Profils textuels (Radar)</b> — compare les deux textes sur six axes stylistiques : diversité lexicale, longueur des phrases, densité, hapax, mots uniques, nombre de phrases.<br>
            <b>Flux (Sankey)</b> — montre les connexions entre segments des deux textes ; la largeur du ruban est proportionnelle au nombre d’alignements.<br>
            <b>Couverture (Timeline)</b> — visualise quelles portions de chaque texte sont couvertes par des alignements et relie les passages correspondants.<br>
            <b>Distribution des scores (Histogramme)</b> — répartition des scores combinés par intervalles de 0,05 avec la moyenne en rouge.<br>
            <b>Densité positionnelle</b> — concentration des alignements selon leur position dans chaque texte ; les pics signalent les zones de fort emprunt.<br>
            <b>Nuage de mots différentiel</b> — vocabulaire exclusif à chaque texte (hors stopwords), taille proportionnelle à la fréquence.<br>
            <b>Matrice de co-occurrence</b> — réseau des termes partagés dans les mêmes passages alignés ; l’épaisseur du lien reflète la force de la co-occurrence.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div class="guide-section">
        <h3>Export CSV</h3>
        <p>L'export des blocs sélectionnés inclut : numéro, positions source/cible, textes (200 car. max)
        et nombre de suppressions/insertions par bloc.</p>
    </div>
    """, unsafe_allow_html=True)

