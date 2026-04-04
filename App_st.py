"""
VERSUS — Interface Streamlit
Alignement sémantique de textes.
"""

import streamlit as st
import streamlit.components.v1 as components
from Corpus import Corpus
from Document import Document
from Comparateur import PairText, FAISS_AVAILABLE
from Global_stuff import Global_stuff
from Config import ComparisonConfig, get_document_hash
from Export import ComparisonExporter
from sentence_transformers import SentenceTransformer
from collections import Counter
import json
import io
import time
import os
import csv
from io import StringIO

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vs_state.json")

st.set_page_config(
    page_title="VERSUS",
    page_icon="logo.gif",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #fafbfc !important; }
    .stApp > header { background-color: #ffffff !important; }
    .stSidebar, .stSidebar > div:first-child, section[data-testid="stSidebar"] { background-color: #ffffff !important; }
    .stMarkdown, .stText, p, span, label { color: #111827 !important; }
    
    .panel-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4b5563 !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .stat-box {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    .stat-value { font-size: 1.75rem; font-weight: 700; color: #2563eb !important; }
    .stat-label { font-size: 0.65rem; text-transform: uppercase; color: #6b7280 !important; }
    
    .stTextInput input, .stTextArea textarea { background-color: #ffffff !important; color: #111827 !important; }
    .stSelectbox > div > div { background-color: #ffffff !important; color: #111827 !important; }
    
    .stFileUploader > div > button { display: none !important; }
    .stFileUploader section > button { display: none !important; }
    [data-testid="stFileUploader"] button { display: none !important; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { display: none !important; }
    
    button[kind="primary"] p,
    button[kind="primary"] span,
    button[kind="primary"] div,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {
        color: white !important;
    }
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span {
        color: #111827 !important;
    }

    .info-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .info-badge-green { background: #d1fae5; color: #065f46; }
    .info-badge-blue { background: #dbeafe; color: #1e40af; }
    .info-badge-gray { background: #f3f4f6; color: #374151; }
    
    .stats-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .stats-table th {
        text-align: left;
        padding: 0.5rem 0.75rem;
        background: #f1f5f9;
        color: #475569 !important;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
    }
    .stats-table td {
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        color: #1e293b !important;
    }
    .stats-table tr:hover td { background: #f8fafc; }
    .freq-bar {
        display: inline-block;
        height: 8px;
        border-radius: 4px;
        margin-right: 6px;
        vertical-align: middle;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def save_state():
    """Sauvegarde l'état minimal de la session dans un fichier JSON local."""
    try:
        data = {
            "align_th":         st.session_state.get("align_th", 0.85),
            "exclude_stopwords":st.session_state.get("exclude_stopwords", False),
            "active_tab":       st.session_state.get("active_tab", 0),
            "align_sort_mode":  st.session_state.get("align_sort_mode", "Combinée (60/40)"),
            "source": (
                {"name": st.session_state.app_source.name,
                 "content": st.session_state.app_source.text.origin_content}
                if st.session_state.get("app_source") else None
            ),
            "corpus": [
                {"name": doc.name, "content": doc.text.origin_content}
                for doc in st.session_state.app_corpus_full.documents
            ] if "app_corpus_full" in st.session_state else [],
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_saved_state():
    """Charge l'état sauvegardé depuis le fichier JSON. Retourne {} si absent."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def init_state():
    if 'app_config' not in st.session_state:
        saved = load_saved_state()
        cfg = ComparisonConfig()
        st.session_state.app_config = cfg
        st.session_state.app_corpus      = Corpus(config=cfg)
        st.session_state.app_corpus_full = Corpus(config=cfg)
        st.session_state.corpus_filter_kw = ''
        st.session_state.app_rankings    = []
        st.session_state.app_target      = None
        st.session_state.app_alignments  = []
        st.session_state.app_comparateur = None
        st.session_state.active_tab      = saved.get("active_tab", 0)
        st.session_state.align_th        = saved.get("align_th", 0.85)
        st.session_state.app_config.similarity_threshold = st.session_state.align_th
        st.session_state.need_realign    = True
        st.session_state.params_pending  = False
        st.session_state.pending_params  = set()
        st.session_state.ranking_pending = False
        st.session_state.exclude_stopwords = saved.get("exclude_stopwords", False)
        st.session_state.app_config.use_stopwords = st.session_state.exclude_stopwords
        st.session_state.removed_corpus_files = set()
        st.session_state.corpus_uploader_key  = 0
        st.session_state.source_uploader_key  = 0
        st.session_state.align_page      = 0
        st.session_state.align_sort_mode = saved.get("align_sort_mode", "Combinée (60/40)")

        # Restaurer la source
        if saved.get("source"):
            try:
                st.session_state.app_source = Document(
                    name=saved["source"]["name"],
                    content=saved["source"]["content"]
                )
            except Exception:
                st.session_state.app_source = None
        else:
            st.session_state.app_source = None

        # Restaurer le corpus
        for doc_data in saved.get("corpus", []):
            try:
                doc = Document(name=doc_data["name"], content=doc_data["content"])
                st.session_state.app_corpus_full.add_doc(doc)
            except Exception:
                pass
        st.session_state.app_corpus = st.session_state.app_corpus_full


def reset_all():
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except Exception:
        pass
    st.session_state.app_config = ComparisonConfig()
    st.session_state.app_corpus = Corpus(config=st.session_state.app_config)
    st.session_state.app_corpus_full = Corpus(config=st.session_state.app_config)
    st.session_state.corpus_filter_kw = ''
    st.session_state.app_source = None
    st.session_state.app_rankings = []
    st.session_state.app_target = None
    st.session_state.app_alignments = []
    st.session_state.app_comparateur = None
    st.session_state.active_tab = 0
    st.session_state.align_th = 0.85
    st.session_state.need_realign = True
    st.session_state.params_pending = False
    st.session_state.pending_params = set()
    st.session_state.ranking_pending = False
    st.session_state.exclude_stopwords = False
    st.session_state.removed_corpus_files = set()
    st.session_state.corpus_uploader_key = st.session_state.get('corpus_uploader_key', 0) + 1
    st.session_state.source_uploader_key = st.session_state.get('source_uploader_key', 0) + 1
    st.session_state.align_page = 0
    st.session_state.align_sort_mode = "Combinée (60/40)"


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.txt'):
        return uploaded_file.getvalue().decode('utf-8', errors='replace')
    elif name.endswith('.docx'):
        try:
            import docx
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            return '\n'.join([p.text for p in doc.paragraphs])
        except:
            return None
    elif name.endswith('.pdf'):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            return '\n'.join([p.extract_text() or '' for p in reader.pages])
        except:
            return None
    return None


def estimate_computation(n_words1, n_words2, config):
    n_sent1 = max(n_words1 // 25, 1)
    n_sent2 = max(n_words2 // 25, 1)
    sentence_time = (n_sent1 + n_sent2) * 0.03
    sentence_mem = (n_sent1 + n_sent2) * 384 * 4 / (1024 * 1024)
    strategy_parts = ["Phrases"]
    if config.ann_enabled and FAISS_AVAILABLE and n_sent2 > 50:
        search_time = n_sent1 * 0.001
        strategy_parts.append("ANN/FAISS")
    else:
        search_time = n_sent1 * n_sent2 * 0.00001
        strategy_parts.append("Exact")
    total_time = max(1, int(sentence_time + search_time))
    total_mem = sentence_mem
    return total_time, round(total_mem, 1), " + ".join(strategy_parts)


def go_to_tab(tab_index):
    st.session_state.active_tab = tab_index
    save_state()


def generate_ranking_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['rang', 'document', 'score', 'mots', 'phrases', 'hash'])
    for r in st.session_state.app_rankings:
        writer.writerow([
            r['rank'], r['doc'].name, f"{r['score']*100:.2f}%",
            len(r['doc'].text.words), r['doc'].text.n_sentences,
            r['doc'].document_hash[:12] if r['doc'].document_hash else "N/A"
        ])
    return output.getvalue()


def generate_alignment_csv():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['num', 'pos_source', 'pos_cible', 'texte_source', 'texte_cible'])
    for i, match in enumerate(st.session_state.app_alignments):
        pos1, pos2 = match[0], match[1]
        t1 = st.session_state.app_source.text.origin_content[pos1[0]:pos1[1]]
        t2 = st.session_state.app_target.text.origin_content[pos2[0]:pos2[1]]
        writer.writerow([
            i + 1, f"{pos1[0]}-{pos1[1]}", f"{pos2[0]}-{pos2[1]}",
            t1[:200], t2[:200]
        ])
    return output.getvalue()


# =================================================================
#  Statistiques textuelles
# =================================================================

def compute_text_stats(text_obj):
    """Calcule les statistiques d'un texte."""
    words = [w.content.lower() for w in text_obj.words]
    sentences = text_obj.sentences
    n_words = len(words)
    n_sentences = len(sentences)
    avg_sent_len = n_words / max(n_sentences, 1)
    word_freq = Counter(words)
    unique_words = len(word_freq)
    return {
        "n_words": n_words,
        "n_sentences": n_sentences,
        "avg_sent_len": round(avg_sent_len, 1),
        "unique_words": unique_words,
        "lexical_diversity": round(unique_words / max(n_words, 1), 3),
        "word_freq": word_freq,
    }


def render_stats(stats1, stats2, name1, name2):
    """Affiche les statistiques comparatives."""
    html = f"""
    <table class="stats-table">
        <tr><th></th><th>{name1}</th><th>{name2}</th></tr>
        <tr><td>Nombre de mots</td><td><b>{stats1['n_words']}</b></td><td><b>{stats2['n_words']}</b></td></tr>
        <tr><td>Nombre de phrases</td><td>{stats1['n_sentences']}</td><td>{stats2['n_sentences']}</td></tr>
        <tr><td>Longueur moyenne des phrases (mots)</td><td>{stats1['avg_sent_len']}</td><td>{stats2['avg_sent_len']}</td></tr>
        <tr><td>Mots uniques</td><td>{stats1['unique_words']}</td><td>{stats2['unique_words']}</td></tr>
        <tr><td>Diversité lexicale</td><td>{stats1['lexical_diversity']}</td><td>{stats2['lexical_diversity']}</td></tr>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)

    common_words = set(stats1['word_freq'].keys()) & set(stats2['word_freq'].keys())
    stopwords = Global_stuff.STOPWORDS
    significant = {w for w in common_words if w not in stopwords and len(w) > 2}

    if significant:
        combined = [(w, stats1['word_freq'][w], stats2['word_freq'][w]) for w in significant]
        combined.sort(key=lambda x: x[1] + x[2], reverse=True)
        top = combined[:10]
        max_freq = max(f1 + f2 for _, f1, f2 in top) if top else 1

        st.markdown("**Fréquences en commun** (hors stopwords, top 10)")
        freq_html = '<table class="stats-table"><tr><th>Mot</th><th>' + name1 + '</th><th>' + name2 + '</th><th></th></tr>'
        for word, f1, f2 in top:
            w1 = int(f1 / max_freq * 100)
            w2 = int(f2 / max_freq * 100)
            freq_html += f'<tr><td><b>{word}</b></td><td>{f1}</td><td>{f2}</td>'
            freq_html += f'<td><span class="freq-bar" style="width:{w1}px;background:#2563eb"></span>'
            freq_html += f'<span class="freq-bar" style="width:{w2}px;background:#f59e0b"></span></td></tr>'
        freq_html += '</table>'
        st.markdown(freq_html, unsafe_allow_html=True)

    only1 = set(stats1['word_freq'].keys()) - set(stats2['word_freq'].keys()) - stopwords
    only2 = set(stats2['word_freq'].keys()) - set(stats1['word_freq'].keys()) - stopwords
    only1_sig = [(w, stats1['word_freq'][w]) for w in only1 if len(w) > 2]
    only2_sig = [(w, stats2['word_freq'][w]) for w in only2 if len(w) > 2]
    only1_sig.sort(key=lambda x: x[1], reverse=True)
    only2_sig.sort(key=lambda x: x[1], reverse=True)

    c1, c2 = st.columns(2)
    with c1:
        if only1_sig:
            st.markdown(f"**Mots propres à {name1}** (top 10)")
            prop_html = '<table class="stats-table"><tr><th>Mot</th><th>Fréq.</th></tr>'
            for w, f in only1_sig[:10]:
                prop_html += f'<tr><td>{w}</td><td>{f}</td></tr>'
            prop_html += '</table>'
            st.markdown(prop_html, unsafe_allow_html=True)
    with c2:
        if only2_sig:
            st.markdown(f"**Mots propres à {name2}** (top 10)")
            prop_html = '<table class="stats-table"><tr><th>Mot</th><th>Fréq.</th></tr>'
            for w, f in only2_sig[:10]:
                prop_html += f'<tr><td>{w}</td><td>{f}</td></tr>'
            prop_html += '</table>'
            st.markdown(prop_html, unsafe_allow_html=True)

# =================================================================
#  Visualisations
# =================================================================

def render_heatmap(alignments, source_text, target_text, name1, name2):
    """Heatmap de densité des alignements (Canvas)."""
    len1 = len(source_text.origin_content)
    len2 = len(target_text.origin_content)
    
    points = []
    for a in alignments:
        pos1, pos2 = a[0], a[1]
        x = ((pos1[0] + pos1[1]) / 2) / max(len1, 1)
        y = ((pos2[0] + pos2[1]) / 2) / max(len2, 1)
        score = float(a[2]) if len(a) > 2 and isinstance(a[2], (int, float)) else 0.5
        seg_len = ((pos1[1] - pos1[0]) + (pos2[1] - pos2[0])) / 2
        r = max(3, min(8, seg_len / 40))
        points.append({"x": round(x, 4), "y": round(y, 4), "s": round(score, 3), "r": round(r, 1)})
    
    points_json = json.dumps(points)
    n1_safe = name1.replace("'", "\\'")
    n2_safe = name2.replace("'", "\\'")
    
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:system-ui,sans-serif;background:#fff}
        .title{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 .5rem}
        canvas{display:block;margin:0 auto}
        .legend{display:flex;justify-content:center;gap:1rem;font-size:.65rem;color:#6b7280;padding:.5rem;flex-wrap:wrap}
        .legend-item{display:flex;align-items:center;gap:4px}
        .legend-dot{width:10px;height:10px;border-radius:50%}
    </style></head><body>
    <div class="title">Carte des alignements</div>
    <canvas id="hm" width="480" height="380"></canvas>
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div> Score faible</div>
        <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div> Score élevé</div>
        <div class="legend-item" style="color:#94a3b8">┈┈ Diagonale</div>
        <div class="legend-item" style="font-weight:600">__N_PTS__ pts</div>
    </div>
    <script>
    const pts=__POINTS__;
    const cv=document.getElementById('hm');
    const ctx=cv.getContext('2d');
    const W=cv.width,H=cv.height;
    const pad={top:20,right:25,bottom:50,left:60};
    const pw=W-pad.left-pad.right;
    const ph=H-pad.top-pad.bottom;
    
    ctx.fillStyle='#f9fafb';
    ctx.fillRect(pad.left,pad.top,pw,ph);
    
    ctx.strokeStyle='#e5e7eb';ctx.lineWidth=0.5;
    for(let i=0;i<=4;i++){
        let x=pad.left+(pw*i/4),y=pad.top+(ph*i/4);
        ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,pad.top+ph);ctx.stroke();
        ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+pw,y);ctx.stroke();
    }
    
    ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.setLineDash([5,5]);
    ctx.beginPath();ctx.moveTo(pad.left,pad.top);ctx.lineTo(pad.left+pw,pad.top+ph);ctx.stroke();
    ctx.setLineDash([]);
    
    for(const p of pts){
        let x=pad.left+p.x*pw,y=pad.top+p.y*ph;
        let t=Math.max(0,Math.min(1,(p.s-0.5)/0.5));
        let r=Math.round(245-t*229),g=Math.round(158+t*27),b=Math.round(11+t*118);
        ctx.fillStyle='rgba('+r+','+g+','+b+',0.7)';
        ctx.beginPath();ctx.arc(x,y,p.r,0,Math.PI*2);ctx.fill();
        ctx.strokeStyle='rgba('+r+','+g+','+b+',0.9)';ctx.lineWidth=0.5;ctx.stroke();
    }
    
    ctx.fillStyle='#6b7280';ctx.font='11px system-ui';ctx.textAlign='center';
    ctx.fillText('__NAME1__ (position →)',pad.left+pw/2,H-8);
    ctx.save();ctx.translate(14,pad.top+ph/2);ctx.rotate(-Math.PI/2);
    ctx.fillText('__NAME2__ (position →)',0,0);ctx.restore();
    
    ctx.fillStyle='#9ca3af';ctx.font='9px system-ui';ctx.textAlign='center';
    for(let i=0;i<=4;i++) ctx.fillText((i*25)+'%',pad.left+pw*i/4,pad.top+ph+15);
    ctx.textAlign='right';
    for(let i=0;i<=4;i++) ctx.fillText((i*25)+'%',pad.left-5,pad.top+ph*i/4+4);
    </script></body></html>"""
    
    html = html.replace('__POINTS__', points_json)
    html = html.replace('__NAME1__', n1_safe)
    html = html.replace('__NAME2__', n2_safe)
    html = html.replace('__N_PTS__', str(len(points)))
    return html


def render_radar(stats1, stats2, name1, name2, stopwords):
    """Radar comparatif des profils textuels (SVG)."""
    import math as _math
    
    # Calcul des métriques supplémentaires
    sw_count1 = sum(f for w, f in stats1['word_freq'].items() if w in stopwords)
    sw_count2 = sum(f for w, f in stats2['word_freq'].items() if w in stopwords)
    density1 = 1 - (sw_count1 / max(stats1['n_words'], 1))
    density2 = 1 - (sw_count2 / max(stats2['n_words'], 1))
    
    hapax1 = sum(1 for w, f in stats1['word_freq'].items() if f == 1) / max(stats1['unique_words'], 1)
    hapax2 = sum(1 for w, f in stats2['word_freq'].items() if f == 1) / max(stats2['unique_words'], 1)
    
    # 6 axes — valeurs brutes
    raw = [
        ("Diversité\nlexicale", stats1['lexical_diversity'], stats2['lexical_diversity']),
        ("Long. moy.\nphrases", stats1['avg_sent_len'], stats2['avg_sent_len']),
        ("Densité\nlexicale", density1, density2),
        ("Hapax", hapax1, hapax2),
        ("Mots\nuniques", stats1['unique_words'], stats2['unique_words']),
        ("Nb\nphrases", stats1['n_sentences'], stats2['n_sentences']),
    ]
    
    # Normalisation 0-1 (par le max des deux)
    axes = []
    for label, v1, v2 in raw:
        mx = max(v1, v2, 0.001)
        axes.append((label, v1 / mx, v2 / mx))
    
    n = len(axes)
    cx, cy, radius = 200, 195, 140
    
    def polar(i, val):
        angle = -_math.pi / 2 + (2 * _math.pi * i / n)
        x = cx + radius * val * _math.cos(angle)
        y = cy + radius * val * _math.sin(angle)
        return round(x, 1), round(y, 1)
    
    # Grille
    grid_svg = ""
    for level in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{polar(i, level)[0]},{polar(i, level)[1]}" for i in range(n))
        grid_svg += f'<polygon points="{pts}" fill="none" stroke="#e5e7eb" stroke-width="0.5"/>\n'
    
    # Axes
    axes_svg = ""
    for i in range(n):
        x, y = polar(i, 1.0)
        axes_svg += f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#d1d5db" stroke-width="0.5"/>\n'
    
    # Labels
    labels_svg = ""
    for i, (label, _, _) in enumerate(axes):
        x, y = polar(i, 1.22)
        lines = label.split('\n')
        for li, line in enumerate(lines):
            ly = y + (li - len(lines)/2 + 0.5) * 12
            labels_svg += f'<text x="{x}" y="{ly}" text-anchor="middle" font-size="10" fill="#6b7280" font-family="system-ui">{line}</text>\n'
    
    # Polygones
    pts1 = " ".join(f"{polar(i, axes[i][1])[0]},{polar(i, axes[i][1])[1]}" for i in range(n))
    pts2 = " ".join(f"{polar(i, axes[i][2])[0]},{polar(i, axes[i][2])[1]}" for i in range(n))
    
    # Points sur les sommets
    dots1 = "".join(f'<circle cx="{polar(i, axes[i][1])[0]}" cy="{polar(i, axes[i][1])[1]}" r="3" fill="#2563eb"/>' for i in range(n))
    dots2 = "".join(f'<circle cx="{polar(i, axes[i][2])[0]}" cy="{polar(i, axes[i][2])[1]}" r="3" fill="#f59e0b"/>' for i in range(n))
    
    n1_safe = name1[:20]
    n2_safe = name2[:20]
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:system-ui,sans-serif;background:#fff}}
        .title{{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 .25rem}}
        .legend{{display:flex;justify-content:center;gap:1.5rem;font-size:.7rem;padding:.5rem}}
        .legend-item{{display:flex;align-items:center;gap:5px}}
        .legend-dot{{width:12px;height:12px;border-radius:3px}}
    </style></head><body>
    <div class="title">Profils textuels comparés</div>
    <svg viewBox="0 0 400 390" width="400" height="390" style="display:block;margin:0 auto">
        {grid_svg}
        {axes_svg}
        <polygon points="{pts1}" fill="rgba(37,99,235,0.15)" stroke="#2563eb" stroke-width="2"/>
        <polygon points="{pts2}" fill="rgba(245,158,11,0.15)" stroke="#f59e0b" stroke-width="2"/>
        {dots1}
        {dots2}
        {labels_svg}
    </svg>
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:rgba(37,99,235,0.3);border:2px solid #2563eb"></div> <b>{n1_safe}</b></div>
        <div class="legend-item"><div class="legend-dot" style="background:rgba(245,158,11,0.3);border:2px solid #f59e0b"></div> <b>{n2_safe}</b></div>
    </div>
    </body></html>"""
    
    return html


def render_sankey(alignments, source_text, target_text, name1, name2, n_segments=5):
    """Graphe de flux (Sankey) source → cible (SVG)."""
    len1 = len(source_text.origin_content)
    len2 = len(target_text.origin_content)
    
    # Compter les alignements entre segments
    flows = {}
    for a in alignments:
        pos1, pos2 = a[0], a[1]
        mid1 = (pos1[0] + pos1[1]) / 2
        mid2 = (pos2[0] + pos2[1]) / 2
        seg1 = min(int(mid1 / max(len1, 1) * n_segments), n_segments - 1)
        seg2 = min(int(mid2 / max(len2, 1) * n_segments), n_segments - 1)
        key = (seg1, seg2)
        flows[key] = flows.get(key, 0) + 1
    
    if not flows:
        return "<div style='text-align:center;color:#9ca3af;padding:2rem;font-family:system-ui'>Aucun flux à afficher</div>"
    
    max_flow = max(flows.values())
    total = sum(flows.values())
    
    # Dimensions SVG
    W, H = 700, 350
    pad_top, pad_bottom = 45, 20
    bar_w = 22
    left_x = 80
    right_x = W - 80
    usable_h = H - pad_top - pad_bottom
    gap = 6
    bar_h = (usable_h - (n_segments - 1) * gap) / n_segments
    
    # Couleurs des segments source
    colors = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
    if n_segments > len(colors):
        colors = colors * ((n_segments // len(colors)) + 1)
    
    svg_bars = ""
    svg_paths = ""
    svg_labels = ""
    
    # Barres source (gauche)
    for i in range(n_segments):
        y = pad_top + i * (bar_h + gap)
        pct_start = round(i * 100 / n_segments)
        pct_end = round((i + 1) * 100 / n_segments)
        svg_bars += f'<rect x="{left_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{colors[i]}"/>\n'
        svg_labels += f'<text x="{left_x - 8}" y="{y + bar_h / 2 + 4}" text-anchor="end" font-size="10" fill="#6b7280" font-family="system-ui">{pct_start}-{pct_end}%</text>\n'
    
    # Barres cible (droite)
    for j in range(n_segments):
        y = pad_top + j * (bar_h + gap)
        pct_start = round(j * 100 / n_segments)
        pct_end = round((j + 1) * 100 / n_segments)
        svg_bars += f'<rect x="{right_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="#f59e0b" opacity="0.8"/>\n'
        svg_labels += f'<text x="{right_x + bar_w + 8}" y="{y + bar_h / 2 + 4}" text-anchor="start" font-size="10" fill="#6b7280" font-family="system-ui">{pct_start}-{pct_end}%</text>\n'
    
    # Flux (courbes de Bézier)
    for (seg1, seg2), count in sorted(flows.items(), key=lambda x: x[1]):
        y1 = pad_top + seg1 * (bar_h + gap) + bar_h / 2
        y2 = pad_top + seg2 * (bar_h + gap) + bar_h / 2
        x1 = left_x + bar_w
        x2 = right_x
        thickness = max(1.5, min(bar_h * 0.8, (count / max_flow) * bar_h * 0.7))
        opacity = max(0.15, min(0.6, count / max_flow * 0.6))
        
        cp_offset = (x2 - x1) * 0.4
        color = colors[seg1]
        
        svg_paths += f'<path d="M{x1},{y1} C{x1 + cp_offset},{y1} {x2 - cp_offset},{y2} {x2},{y2}" fill="none" stroke="{color}" stroke-width="{round(thickness, 1)}" opacity="{round(opacity, 2)}"/>\n'
        
        # Étiquette au milieu du flux si significatif
        if count >= max(2, total * 0.05):
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            svg_paths += f'<text x="{mx}" y="{my - 2}" text-anchor="middle" font-size="9" fill="#374151" font-weight="600" font-family="system-ui">{count}</text>\n'
    
    n1_safe = name1[:25]
    n2_safe = name2[:25]
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:system-ui,sans-serif;background:#fff}}
        .title{{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 0}}
        .subtitle{{font-size:.65rem;color:#9ca3af;text-align:center;padding:.25rem 0 .5rem}}
    </style></head><body>
    <div class="title">Flux des alignements</div>
    <div class="subtitle">{total} alignements entre {n_segments} segments</div>
    <svg viewBox="0 0 {W} {H}" width="100%" height="{H}" style="display:block">
        {svg_paths}
        {svg_bars}
        {svg_labels}
        <text x="{left_x + bar_w / 2}" y="{pad_top - 12}" text-anchor="middle" font-size="11" font-weight="700" fill="#2563eb" font-family="system-ui">{n1_safe}</text>
        <text x="{right_x + bar_w / 2}" y="{pad_top - 12}" text-anchor="middle" font-size="11" font-weight="700" fill="#f59e0b" font-family="system-ui">{n2_safe}</text>
    </svg>
    </body></html>"""
    
    return html


def render_timeline(alignments, source_text, target_text, name1, name2):
    """Timeline parallèle : couverture des alignements sur les deux textes."""
    len1 = len(source_text.origin_content)
    len2 = len(target_text.origin_content)
    
    # Collecter les zones alignées
    src_zones = []
    tgt_zones = []
    links = []
    for a in alignments:
        pos1, pos2 = a[0], a[1]
        s1 = pos1[0] / max(len1, 1)
        e1 = pos1[1] / max(len1, 1)
        s2 = pos2[0] / max(len2, 1)
        e2 = pos2[1] / max(len2, 1)
        src_zones.append((s1, e1))
        tgt_zones.append((s2, e2))
        links.append((s1, e1, s2, e2))
    
    # Calculer couverture
    def coverage(zones):
        if not zones:
            return 0
        merged = []
        for s, e in sorted(zones):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return sum(e - s for s, e in merged)
    
    cov1 = round(coverage(src_zones) * 100, 1)
    cov2 = round(coverage(tgt_zones) * 100, 1)
    
    W, H = 720, 200
    bar_h = 24
    y1 = 55
    y2 = 135
    pad_l, pad_r = 70, 30
    bw = W - pad_l - pad_r
    
    n1_safe = name1[:30].replace("'", "\\'")
    n2_safe = name2[:30].replace("'", "\\'")
    
    # Barres de fond
    svg = f'<rect x="{pad_l}" y="{y1}" width="{bw}" height="{bar_h}" rx="4" fill="#f1f5f9" stroke="#e2e8f0"/>\n'
    svg += f'<rect x="{pad_l}" y="{y2}" width="{bw}" height="{bar_h}" rx="4" fill="#f1f5f9" stroke="#e2e8f0"/>\n'
    
    # Lignes de connexion (d'abord, pour qu'elles soient derrière)
    link_svg = ""
    max_links = min(len(links), 200)  # limiter pour perf
    step = max(1, len(links) // max_links)
    for i in range(0, len(links), step):
        s1, e1, s2, e2 = links[i]
        mx1 = pad_l + ((s1 + e1) / 2) * bw
        mx2 = pad_l + ((s2 + e2) / 2) * bw
        link_svg += f'<line x1="{round(mx1,1)}" y1="{y1 + bar_h}" x2="{round(mx2,1)}" y2="{y2}" stroke="#93c5fd" stroke-width="0.8" opacity="0.3"/>\n'
    
    # Zones source
    for s, e in src_zones:
        x = pad_l + s * bw
        w = max(1, (e - s) * bw)
        svg += f'<rect x="{round(x,1)}" y="{y1}" width="{round(w,1)}" height="{bar_h}" fill="#2563eb" opacity="0.6" rx="2"/>\n'
    
    # Zones cible
    for s, e in tgt_zones:
        x = pad_l + s * bw
        w = max(1, (e - s) * bw)
        svg += f'<rect x="{round(x,1)}" y="{y2}" width="{round(w,1)}" height="{bar_h}" fill="#f59e0b" opacity="0.6" rx="2"/>\n'
    
    # Labels
    labels = f'<text x="{pad_l - 8}" y="{y1 + bar_h/2 + 4}" text-anchor="end" font-size="10" font-weight="600" fill="#2563eb" font-family="system-ui">{n1_safe}</text>\n'
    labels += f'<text x="{pad_l - 8}" y="{y2 + bar_h/2 + 4}" text-anchor="end" font-size="10" font-weight="600" fill="#f59e0b" font-family="system-ui">{n2_safe}</text>\n'
    
    # Graduations
    grads = ""
    for i in range(0, 101, 25):
        x = pad_l + (i / 100) * bw
        grads += f'<text x="{round(x)}" y="{y1 - 8}" text-anchor="middle" font-size="8" fill="#9ca3af" font-family="system-ui">{i}%</text>\n'
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:system-ui,sans-serif;background:#fff}}
        .title{{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 0}}
        .subtitle{{font-size:.65rem;color:#9ca3af;text-align:center;padding:.25rem 0 .5rem}}
    </style></head><body>
    <div class="title">Couverture des alignements</div>
    <div class="subtitle">Source : {cov1}% couvert · Cible : {cov2}% couvert · {len(alignments)} alignements</div>
    <svg viewBox="0 0 {W} {H}" width="100%" height="{H}" style="display:block">
        {grads}
        {link_svg}
        {svg}
        {labels}
    </svg>
    </body></html>"""
    
    return html


def render_wordcloud(stats1, stats2, name1, name2, stopwords):
    """Nuage de mots différentiel (HTML/CSS)."""
    freq1 = stats1['word_freq']
    freq2 = stats2['word_freq']
    
    # Mots propres à chaque texte
    only1 = {w: f for w, f in freq1.items() if w not in freq2 and w not in stopwords and len(w) > 2}
    only2 = {w: f for w, f in freq2.items() if w not in freq1 and w not in stopwords and len(w) > 2}
    
    # Mots partagés
    shared_words = {w for w in freq1 if w in freq2 and w not in stopwords and len(w) > 2}
    shared = {}
    for w in shared_words:
        f1, f2 = freq1[w], freq2[w]
        shared[w] = (f1, f2, f1 + f2)
    
    def top_n(d, n=25):
        return sorted(d.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1][2], reverse=True)[:n]
    
    top_only1 = top_n(only1, 20)
    top_only2 = top_n(only2, 20)
    top_shared = top_n(shared, 25)
    
    # Max freq pour scaling
    all_freqs = [f for _, f in top_only1] + [f for _, f in top_only2] + [t[2] for _, t in top_shared]
    max_f = max(all_freqs) if all_freqs else 1
    
    def size(f, mn=11, mx=28):
        return round(mn + (f / max_f) * (mx - mn), 1)
    
    def make_cloud(items, color_fn):
        spans = []
        for w, f in items:
            freq_val = f if isinstance(f, (int, float)) else f[2]
            s = size(freq_val)
            c = color_fn(w, f)
            spans.append(f'<span style="font-size:{s}px;color:{c};padding:2px 5px;display:inline-block;font-weight:{500 if s < 18 else 700}">{w}</span>')
        return " ".join(spans)
    
    cloud1 = make_cloud(top_only1, lambda w, f: "#2563eb")
    cloud2 = make_cloud(top_only2, lambda w, f: "#d97706")
    
    def shared_color(w, f):
        f1, f2, total = f
        ratio = f1 / max(f1 + f2, 1)
        if ratio > 0.6:
            return "#3b82f6"
        elif ratio < 0.4:
            return "#f59e0b"
        else:
            return "#6b7280"
    
    cloud_shared = make_cloud(top_shared, shared_color)
    
    n1 = name1[:20]
    n2 = name2[:20]
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:Georgia,serif;background:#fff}}
        .title{{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 .5rem;font-family:system-ui}}
        .grid{{display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:8px;padding:0 12px 12px}}
        .panel{{background:#fafbfc;border:1px solid #e5e7eb;border-radius:8px;padding:10px;text-align:center;min-height:120px;line-height:2}}
        .label{{font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;font-family:system-ui}}
        .count{{font-size:.6rem;color:#9ca3af;font-family:system-ui}}
    </style></head><body>
    <div class="title">Vocabulaire différentiel</div>
    <div class="grid">
        <div class="panel">
            <div class="label" style="color:#2563eb">Propres à {n1}</div>
            {cloud1}
            <div class="count">{len(only1)} mots uniques</div>
        </div>
        <div class="panel" style="border-color:#d1d5db">
            <div class="label" style="color:#374151">Partagés</div>
            {cloud_shared}
            <div class="count">{len(shared)} mots · <span style="color:#3b82f6">■</span> {n1} dominant · <span style="color:#f59e0b">■</span> {n2} dominant</div>
        </div>
        <div class="panel">
            <div class="label" style="color:#d97706">Propres à {n2}</div>
            {cloud2}
            <div class="count">{len(only2)} mots uniques</div>
        </div>
    </div>
    </body></html>"""
    
    return html


def render_score_histogram(alignments):
    """Histogramme de distribution des scores de similarité (Canvas)."""
    # Extraire les scores (certains alignements peuvent ne pas avoir de score)
    scores = []
    for a in alignments:
        if len(a) > 2 and isinstance(a[2], (int, float)):
            scores.append(float(a[2]))
    
    if not scores:
        return "<div style='text-align:center;color:#9ca3af;padding:2rem;font-family:system-ui'>Aucun score disponible</div>"
    
    # Bins de 0.05
    bins = {}
    for s in scores:
        b = round(int(s / 0.05) * 0.05, 2)
        b = max(0.5, min(b, 0.95))
        bins[b] = bins.get(b, 0) + 1
    
    all_bins = []
    for i in range(10, 21):
        val = round(i * 0.05, 2)
        all_bins.append((val, bins.get(val, 0)))
    
    max_count = max(c for _, c in all_bins) if all_bins else 1
    avg_score = sum(scores) / len(scores)
    med_score = sorted(scores)[len(scores) // 2]
    
    bins_json = json.dumps(all_bins)
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:system-ui,sans-serif;background:#fff}}
        .title{{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 0}}
        .subtitle{{font-size:.65rem;color:#9ca3af;text-align:center;padding:.25rem 0 .5rem}}
        canvas{{display:block;margin:0 auto}}
    </style></head><body>
    <div class="title">Distribution des scores</div>
    <div class="subtitle">{len(scores)} scores · Moy: {avg_score:.3f} · Méd: {med_score:.3f}</div>
    <canvas id="hist" width="440" height="250"></canvas>
    <script>
    const bins={bins_json};
    const maxC={max_count};
    const avg={round(avg_score, 4)};
    const cv=document.getElementById('hist');
    const ctx=cv.getContext('2d');
    const W=cv.width,H=cv.height;
    const pad={{top:15,right:20,bottom:45,left:50}};
    const pw=W-pad.left-pad.right;
    const ph=H-pad.top-pad.bottom;
    
    ctx.fillStyle='#f9fafb';
    ctx.fillRect(pad.left,pad.top,pw,ph);
    
    // Grille
    ctx.strokeStyle='#f1f5f9';ctx.lineWidth=0.5;
    for(let i=0;i<=4;i++){{
        let y=pad.top+ph*(1-i/4);
        ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+pw,y);ctx.stroke();
    }}
    
    // Barres
    const bw=pw/bins.length-4;
    for(let i=0;i<bins.length;i++){{
        let [val,count]=bins[i];
        let x=pad.left+i*(bw+4)+2;
        let h=count/maxC*ph;
        let y=pad.top+ph-h;
        
        let t=Math.max(0,Math.min(1,(val-0.5)/0.5));
        let r=Math.round(37+t*0),g=Math.round(99+t*86),b_c=Math.round(235-t*106);
        ctx.fillStyle='rgb('+r+','+g+','+b_c+')';
        ctx.beginPath();
        ctx.roundRect(x,y,bw,h,3);
        ctx.fill();
        
        // Label X
        ctx.fillStyle='#9ca3af';ctx.font='9px system-ui';ctx.textAlign='center';
        ctx.fillText(val.toFixed(2),x+bw/2,pad.top+ph+13);
        
        // Count
        if(count>0){{
            ctx.fillStyle='#374151';ctx.font='bold 9px system-ui';
            ctx.fillText(count,x+bw/2,y-4);
        }}
    }}
    
    // Ligne moyenne
    let avgX=pad.left+((avg-0.5)/0.55)*pw;
    ctx.strokeStyle='#ef4444';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(avgX,pad.top);ctx.lineTo(avgX,pad.top+ph);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#ef4444';ctx.font='bold 9px system-ui';ctx.textAlign='center';
    ctx.fillText('moy.',avgX,pad.top+ph+28);
    
    // Axe Y
    ctx.fillStyle='#9ca3af';ctx.font='9px system-ui';ctx.textAlign='right';
    for(let i=0;i<=4;i++){{
        let v=Math.round(maxC*i/4);
        let y=pad.top+ph*(1-i/4);
        ctx.fillText(v,pad.left-5,y+3);
    }}
    </script></body></html>"""
    
    return html


def render_density(alignments, source_text, target_text, name1, name2, n_bins=30):
    """Courbe de densité positionnelle des alignements (Canvas)."""
    len1 = len(source_text.origin_content)
    len2 = len(target_text.origin_content)
    
    density1 = [0] * n_bins
    density2 = [0] * n_bins
    
    for a in alignments:
        pos1, pos2 = a[0], a[1]
        mid1 = (pos1[0] + pos1[1]) / 2
        mid2 = (pos2[0] + pos2[1]) / 2
        b1 = min(int(mid1 / max(len1, 1) * n_bins), n_bins - 1)
        b2 = min(int(mid2 / max(len2, 1) * n_bins), n_bins - 1)
        density1[b1] += 1
        density2[b2] += 1
    
    max_d = max(max(density1), max(density2), 1)
    
    d1_json = json.dumps(density1)
    d2_json = json.dumps(density2)
    n1_safe = name1[:20].replace("'", "\\'")
    n2_safe = name2[:20].replace("'", "\\'")
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:system-ui,sans-serif;background:#fff}}
        .title{{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 0}}
        .subtitle{{font-size:.65rem;color:#9ca3af;text-align:center;padding:.25rem 0 .5rem}}
        canvas{{display:block;margin:0 auto}}
        .legend{{display:flex;justify-content:center;gap:1.5rem;font-size:.7rem;padding:.5rem}}
        .legend-item{{display:flex;align-items:center;gap:5px}}
        .legend-line{{width:20px;height:3px;border-radius:2px}}
    </style></head><body>
    <div class="title">Densité des alignements par position</div>
    <div class="subtitle">{n_bins} segments · max {max_d} alignements/segment</div>
    <canvas id="dens" width="480" height="230"></canvas>
    <div class="legend">
        <div class="legend-item"><div class="legend-line" style="background:#2563eb"></div> {n1_safe}</div>
        <div class="legend-item"><div class="legend-line" style="background:#f59e0b"></div> {n2_safe}</div>
    </div>
    <script>
    const d1={d1_json},d2={d2_json};
    const maxD={max_d};
    const cv=document.getElementById('dens');
    const ctx=cv.getContext('2d');
    const W=cv.width,H=cv.height;
    const pad={{top:15,right:20,bottom:35,left:45}};
    const pw=W-pad.left-pad.right;
    const ph=H-pad.top-pad.bottom;
    
    ctx.fillStyle='#f9fafb';
    ctx.fillRect(pad.left,pad.top,pw,ph);
    
    // Grille
    ctx.strokeStyle='#f1f5f9';ctx.lineWidth=0.5;
    for(let i=0;i<=4;i++){{
        let y=pad.top+ph*(1-i/4);
        ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+pw,y);ctx.stroke();
    }}
    
    function drawCurve(data,color,fillColor){{
        let n=data.length;
        ctx.beginPath();
        ctx.moveTo(pad.left,pad.top+ph);
        for(let i=0;i<n;i++){{
            let x=pad.left+(i+0.5)/n*pw;
            let y=pad.top+ph*(1-data[i]/maxD);
            if(i===0) ctx.lineTo(x,y);
            else{{
                let px=pad.left+(i-0.5)/n*pw;
                let cpx=(px+x)/2;
                ctx.bezierCurveTo(cpx,pad.top+ph*(1-data[i-1]/maxD),cpx,y,x,y);
            }}
        }}
        ctx.lineTo(pad.left+pw,pad.top+ph);
        ctx.closePath();
        ctx.fillStyle=fillColor;
        ctx.fill();
        
        ctx.beginPath();
        for(let i=0;i<n;i++){{
            let x=pad.left+(i+0.5)/n*pw;
            let y=pad.top+ph*(1-data[i]/maxD);
            if(i===0) ctx.moveTo(x,y);
            else{{
                let px=pad.left+(i-0.5)/n*pw;
                let cpx=(px+x)/2;
                ctx.bezierCurveTo(cpx,pad.top+ph*(1-data[i-1]/maxD),cpx,y,x,y);
            }}
        }}
        ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.stroke();
    }}
    
    drawCurve(d1,'#2563eb','rgba(37,99,235,0.1)');
    drawCurve(d2,'#f59e0b','rgba(245,158,11,0.1)');
    
    // Axes
    ctx.fillStyle='#9ca3af';ctx.font='9px system-ui';
    ctx.textAlign='center';
    for(let i=0;i<=4;i++) ctx.fillText((i*25)+'%',pad.left+pw*i/4,pad.top+ph+15);
    ctx.textAlign='right';
    for(let i=0;i<=4;i++){{
        let v=Math.round(maxD*i/4);
        ctx.fillText(v,pad.left-5,pad.top+ph*(1-i/4)+4);
    }}
    ctx.textAlign='center';
    ctx.fillText('Position dans le texte (%)',pad.left+pw/2,pad.top+ph+28);
    </script></body></html>"""
    
    return html


def render_cooccurrence(alignments, source_text, target_text, stopwords, n_top=12):
    """Matrice de co-occurrence des mots-clés dans les segments alignés (Canvas)."""
    from collections import Counter as _Counter
    import re as _re
    
    # Extraire les mots significatifs de chaque segment aligné
    word_segments = []
    global_freq = _Counter()
    
    for a in alignments:
        pos1, pos2 = a[0], a[1]
        t1 = source_text.origin_content[pos1[0]:pos1[1]]
        t2 = target_text.origin_content[pos2[0]:pos2[1]]
        combined = t1 + " " + t2
        words = set(w.lower() for w in _re.findall(r'\b\w+\b', combined) if len(w) > 2 and w.lower() not in stopwords)
        word_segments.append(words)
        for w in words:
            global_freq[w] += 1
    
    # Top N mots les plus fréquents
    top_words = [w for w, _ in global_freq.most_common(n_top)]
    n = len(top_words)
    
    if n < 3:
        return "<div style='text-align:center;color:#9ca3af;padding:2rem;font-family:system-ui'>Pas assez de mots-clés pour la matrice</div>"
    
    # Matrice de co-occurrence
    matrix = [[0] * n for _ in range(n)]
    for seg_words in word_segments:
        present = [i for i in range(n) if top_words[i] in seg_words]
        for a in present:
            for b in present:
                if a != b:
                    matrix[a][b] += 1
    
    max_val = max(matrix[i][j] for i in range(n) for j in range(n) if i != j) if n > 1 else 1
    max_val = max(max_val, 1)
    
    matrix_json = json.dumps(matrix)
    words_json = json.dumps(top_words)
    
    cell_size = max(28, min(42, 420 // n))
    W = 100 + n * cell_size + 20
    H = 80 + n * cell_size + 20
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:system-ui,sans-serif;background:#fff}}
        .title{{font-size:.75rem;font-weight:600;color:#374151;text-transform:uppercase;letter-spacing:.05em;text-align:center;padding:.75rem 0 0}}
        .subtitle{{font-size:.65rem;color:#9ca3af;text-align:center;padding:.25rem 0 .5rem}}
        canvas{{display:block;margin:0 auto}}
    </style></head><body>
    <div class="title">Co-occurrence des mots-clés</div>
    <div class="subtitle">{n} mots les plus fréquents dans les {len(alignments)} segments alignés</div>
    <canvas id="cooc" width="{W}" height="{H}"></canvas>
    <script>
    const m={matrix_json};
    const words={words_json};
    const maxV={max_val};
    const n={n};
    const cs={cell_size};
    const cv=document.getElementById('cooc');
    const ctx=cv.getContext('2d');
    const ox=95,oy=70;
    
    // Cellules
    for(let i=0;i<n;i++){{
        for(let j=0;j<n;j++){{
            let x=ox+j*cs,y=oy+i*cs;
            if(i===j){{
                ctx.fillStyle='#f1f5f9';
            }} else {{
                let t=m[i][j]/maxV;
                let r=Math.round(255-t*218);
                let g=Math.round(255-t*156);
                let b_c=Math.round(255-t*20);
                ctx.fillStyle='rgb('+r+','+g+','+b_c+')';
            }}
            ctx.fillRect(x,y,cs-1,cs-1);
            
            if(i!==j && m[i][j]>0){{
                let t=m[i][j]/maxV;
                ctx.fillStyle=t>0.5?'#fff':'#374151';
                ctx.font='bold '+(cs>35?'11':'9')+'px system-ui';
                ctx.textAlign='center';
                ctx.fillText(m[i][j],x+cs/2-0.5,y+cs/2+3.5);
            }}
        }}
    }}
    
    // Labels
    ctx.fillStyle='#374151';
    ctx.font='10px system-ui';
    ctx.textAlign='right';
    for(let i=0;i<n;i++){{
        let label=words[i].length>10?words[i].substring(0,9)+'…':words[i];
        ctx.fillText(label,ox-5,oy+i*cs+cs/2+3);
    }}
    
    ctx.textAlign='center';
    for(let j=0;j<n;j++){{
        ctx.save();
        ctx.translate(ox+j*cs+cs/2,oy-5);
        ctx.rotate(-Math.PI/4);
        let label=words[j].length>10?words[j].substring(0,9)+'…':words[j];
        ctx.fillText(label,0,0);
        ctx.restore();
    }}
    </script></body></html>"""
    
    return html


# =================================================================
#  Main
# =================================================================

def main():
    init_state()
    
    # === SIDEBAR ===
    with st.sidebar:
        import base64
        logo_path = os.path.join(os.path.dirname(__file__), "logo2.gif")
        logo_img = ''
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="height:2rem">'
        
        # Bouton caché (sera cliqué par le JS du logo)
        if st.button("___versus_reset___", key="reset_btn"):
            reset_all()
            st.rerun()
        
        # Logo + titre cliquable via components.html (+ cache le bouton reset par JS)
        components.html(f"""
        <div onclick="
            var btns = window.parent.document.querySelectorAll('button');
            for(var i=0;i<btns.length;i++){{
                if(btns[i].innerText.indexOf('versus_reset')!==-1){{btns[i].click();break;}}
            }}
        " style="display:flex;align-items:center;justify-content:center;gap:10px;padding:0.5rem 0;cursor:pointer;border-radius:8px;transition:background 0.2s;user-select:none"
           onmouseover="this.style.background='#f3f4f6'" onmouseout="this.style.background='transparent'">
            {logo_img}
            <span style="font-size:1.8rem;font-weight:800;letter-spacing:0.05em;color:#111827;font-family:system-ui,sans-serif">VERSUS</span>
        </div>
        <script>
            // Cacher le bouton reset par son texte (fiable quelle que soit la version Streamlit)
            var btns = window.parent.document.querySelectorAll('button');
            for(var i=0;i<btns.length;i++){{
                if(btns[i].innerText.indexOf('versus_reset')!==-1){{
                    var el=btns[i].closest('[data-testid="stButton"]')||btns[i].parentElement;
                    el.style.cssText='height:0;overflow:hidden;margin:0;padding:0;position:absolute';
                    break;
                }}
            }}
        </script>
        """, height=55)

        # ── Vignettes paramètres ──
        st.divider()
        def _pill(label, color, bg):
            return (f"<span style='background:{bg};color:{color};padding:2px 8px;"
                    f"border-radius:12px;font-size:0.7rem;font-weight:600;"
                    f"margin:2px 2px;display:inline-block'>{label}</span>")
        cfg_b = st.session_state.app_config
        sw_val = st.session_state.get('exclude_stopwords', False)
        th_b = st.session_state.align_th
        th_c = ("#065f46","#d1fae5") if th_b >= 0.85 else ("#92400e","#fef3c7") if th_b >= 0.70 else ("#991b1b","#fee2e2")
        pills = ""
        pills += _pill("−SW" if sw_val else "+SW", "#065f46" if sw_val else "#6b7280", "#d1fae5" if sw_val else "#f3f4f6")
        pills += _pill(f"seuil {th_b:.2f}", th_c[0], th_c[1])
        st.markdown(f"<div style='line-height:2'>{pills}</div>", unsafe_allow_html=True)

        cfg = st.session_state.app_config

        st.divider()

        _show_w = bool(st.session_state.app_alignments)
        _w = "⚠️ Relancez l'alignement pour appliquer."

        # 1. Stopwords
        new_exclude_sw = st.checkbox("Exclure les stopwords", st.session_state.exclude_stopwords,
                                     key="cb_exclude_sw",
                                     help="Les stopwords sont des mots grammaticaux très fréquents (articles, prépositions, conjonctions…) qui n'apportent pas de sens propre. Les exclure concentre la comparaison sur le vocabulaire significatif, ce qui améliore la précision pour des textes à contenu dense. À désactiver si la structure grammaticale elle-même est pertinente pour l'analyse.")
        if new_exclude_sw != st.session_state.exclude_stopwords:
            st.session_state.exclude_stopwords = new_exclude_sw
            st.session_state.app_config.use_stopwords = new_exclude_sw
            st.session_state.params_pending = True
            st.session_state.pending_params.add("sw")
            save_state()
        if "sw" in st.session_state.pending_params and _show_w:
            st.warning(_w)

        st.divider()

        # 2. Seuil cosinus
        st.markdown("Seuil cosinus", help="Score minimum de similarité cosinus qu'une paire de segments doit atteindre pour être retenue. C'est le paramètre qui a le plus d'impact sur le volume de résultats. La comparaison des embeddings est faite par cosinus (ou via FAISS si le texte cible dépasse 50 phrases).\n\nUn seuil élevé (> 0,85) ne garde que les correspondances très proches ; un seuil bas (< 0,70) détecte plus de rapprochements mais introduit davantage de faux positifs.")
        new_th = st.slider(
            "Seuil cosinus", 0.5, 1.0, st.session_state.align_th, 0.01,
            key="slider_th",
            label_visibility="collapsed"
        )
        if abs(new_th - st.session_state.align_th) > 0.001:
            st.session_state.align_th = new_th
            st.session_state.params_pending = True
            st.session_state.pending_params.add("th")
            save_state()
        if "th" in st.session_state.pending_params and _show_w:
            st.warning(_w)

        st.session_state.app_config.adaptive_threshold = True
    
    # === NAVIGATION ===
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("1 - Chargement", use_container_width=True, 
                    type="primary" if st.session_state.active_tab == 0 else "secondary"):
            go_to_tab(0); st.rerun()
    with col2:
        if st.button("2 - Traitement", use_container_width=True,
                    type="primary" if st.session_state.active_tab == 1 else "secondary"):
            go_to_tab(1); st.rerun()
    with col3:
        if st.button("3 - Résultats", use_container_width=True,
                    type="primary" if st.session_state.active_tab == 2 else "secondary"):
            go_to_tab(2); st.rerun()
    with col4:
        if st.button("Stats et Viz", use_container_width=True,
                    type="primary" if st.session_state.active_tab == 4 else "secondary"):
            go_to_tab(4); st.rerun()
    with col5:
        if st.button("Guide", use_container_width=True,
                    type="primary" if st.session_state.active_tab == 3 else "secondary"):
            go_to_tab(3); st.rerun()
    
    st.divider()

    # ===================== PAGE 1) ENTRÉE =====================
    if st.session_state.active_tab == 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="panel-title">📄 TEXTE SOURCE</div>', unsafe_allow_html=True)
            src_file = st.file_uploader(
                "Glisser-déposer", type=['txt', 'docx', 'pdf'],
                key=f"src_upload_{st.session_state.source_uploader_key}",
                label_visibility="collapsed"
            )
            if src_file:
                # Ne recréer que si c'est un nouveau fichier
                if st.session_state.app_source is None or st.session_state.app_source.name != src_file.name:
                    content = read_file(src_file)
                    if content:
                        st.session_state.app_source = Document(name=src_file.name, content=content)
                        save_state()
            
            if st.session_state.app_source:
                doc = st.session_state.app_source
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"📄 {doc.name} ({len(doc.text.words)} mots)")
                with c2:
                    if st.button("✕", key="rm_src"):
                        st.session_state.app_source = None
                        st.session_state.source_uploader_key += 1
                        st.rerun()
        
        with col2:
            st.markdown('<div class="panel-title">📚 CORPUS CIBLE</div>', unsafe_allow_html=True)
            corpus_files = st.file_uploader(
                "Glisser-déposer", type=['txt', 'docx', 'pdf'], 
                accept_multiple_files=True, 
                key=f"corpus_upload_{st.session_state.corpus_uploader_key}", 
                label_visibility="collapsed"
            )
            if corpus_files:
                added = 0
                for f in corpus_files:
                    existing_full = st.session_state.app_corpus_full.get_documents_names()
                    file_content = read_file(f)
                    if not file_content:
                        continue
                    if f.name not in existing_full:
                        new_doc = Document(name=f.name, content=file_content)
                        st.session_state.app_corpus_full.add_doc(new_doc)
                        added += 1
                    else:
                        from Config import get_document_hash
                        new_hash = get_document_hash(file_content)
                        idx = existing_full.index(f.name)
                        if new_hash != st.session_state.app_corpus_full.documents[idx].document_hash:
                            st.session_state.app_corpus_full.documents[idx] = Document(name=f.name, content=file_content)
                            st.session_state.app_corpus_full.corpus_weights = None
                            if st.session_state.app_target and st.session_state.app_target.name == f.name:
                                st.session_state.app_target = None
                                st.session_state.app_alignments = []
                            st.session_state.app_rankings = []
                            st.session_state.need_realign = True
                            added += 1
                if added > 0:
                    # Re-appliquer le filtre actif s'il y en a un
                    if st.session_state.corpus_filter_kw:
                        st.session_state.app_corpus = st.session_state.app_corpus_full.filter(
                            st.session_state.corpus_filter_kw, ignore_case=True)
                    else:
                        st.session_state.app_corpus = st.session_state.app_corpus_full
                    st.session_state.app_rankings = []
                    st.session_state.need_realign = True
                    save_state()
                    st.rerun()
            
            # Toujours afficher TOUS les fichiers du corpus complet dans le drag & drop
            if len(st.session_state.app_corpus_full) > 0:
                for i, doc in enumerate(st.session_state.app_corpus_full.documents):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.caption(f"📄 {doc.name} ({len(doc.text.words)} mots)")
                    with c2:
                        if st.button("✕", key=f"rm_{i}"):
                            removed_name = st.session_state.app_corpus_full.documents[i].name
                            # Supprimer du corpus complet
                            st.session_state.app_corpus_full = st.session_state.app_corpus_full - i
                            # Re-appliquer le filtre actif si besoin
                            if st.session_state.corpus_filter_kw:
                                st.session_state.app_corpus = st.session_state.app_corpus_full.filter(
                                    st.session_state.corpus_filter_kw, ignore_case=True)
                            else:
                                st.session_state.app_corpus = st.session_state.app_corpus_full
                            st.session_state.corpus_uploader_key += 1
                            st.session_state.app_rankings = []
                            st.session_state.need_realign = True
                            if st.session_state.app_target and st.session_state.app_target.name == removed_name:
                                st.session_state.app_target = None
                                st.session_state.app_alignments = []
                            st.rerun()
        
        # Filtre mots-clés
        st.divider()
        active_filter = st.session_state.corpus_filter_kw
        st.markdown('<div class="panel-title">🔎 FILTRER LE CORPUS CIBLE PAR MOTS-CLÉS</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            kw = st.text_input("Mot-clé", label_visibility="collapsed", placeholder="Ex: philosophie")
        with c2:
            case = st.checkbox("Ignorer casse", True)
        with c3:
            if st.button("Filtrer", use_container_width=True) and kw:
                st.session_state.corpus_filter_kw = kw
                st.session_state.app_corpus = st.session_state.app_corpus_full.filter(kw, ignore_case=case)
                st.rerun()
        with c4:
            if st.button("↺ Reset", use_container_width=True, disabled=not active_filter):
                st.session_state.corpus_filter_kw = ""
                st.session_state.app_corpus = st.session_state.app_corpus_full
                st.rerun()

        # Résultats du filtre (Option B)
        if active_filter:
            filtered_docs = st.session_state.app_corpus.documents
            total_docs = len(st.session_state.app_corpus_full)
            n_found = len(filtered_docs)
            if n_found == 0:
                st.warning(f"Aucun document cible ne contient « {active_filter} ».")
            else:
                st.markdown(
                    f"<div style='font-size:0.78rem;color:#6b7280;margin:0.4rem 0 0.6rem 0'>"
                    f"<b style='color:#2563eb'>{n_found}</b> document(s) retenu(s) sur "
                    f"<b>{total_docs}</b> — filtre : <b>« {active_filter} »</b></div>",
                    unsafe_allow_html=True
                )
                for doc in filtered_docs:
                    matches = st.session_state.app_corpus.keyword_matches.get(doc.name, 0)
                    st.markdown(
                        f"<div style='padding:0.35rem 0.75rem;margin:0.2rem 0;"
                        f"background:#f0f7ff;border-left:3px solid #2563eb;"
                        f"border-radius:4px;font-size:0.8rem;'>"
                        f"📄 <b>{doc.name}</b> "
                        f"<span style='color:#6b7280'>({len(doc.text.words)} mots)</span> "
                        f"<span style='color:#2563eb;font-weight:600;margin-left:0.5rem'>"
                        f"{matches} occurrence(s)</span></div>",
                        unsafe_allow_html=True
                    )

        can_classify = st.session_state.app_source is not None and len(st.session_state.app_corpus) > 0
        if st.button("Valider →", use_container_width=True, disabled=not can_classify, type="primary"):
            st.session_state.active_tab = 1
            st.rerun()
    
    # ===================== PAGE 2) CLASSEMENT =====================
    elif st.session_state.active_tab == 1:
        if not st.session_state.app_source:
            st.warning("⚠️ Chargez un texte source dans l'onglet Entrée")
        elif len(st.session_state.app_corpus) == 0:
            st.warning("⚠️ Ajoutez des documents au corpus dans l'onglet Entrée")
        else:
            already_ranked = bool(st.session_state.app_rankings)

            if st.button("Classer les textes cibles par similarité →", use_container_width=True,
                         type="secondary" if already_ranked else "primary"):
                with st.spinner("Classement en cours..."):
                    temp = st.session_state.app_corpus.copy()
                    idx = temp.index(st.session_state.app_source)
                    if idx == -1:
                        temp.add_doc(st.session_state.app_source)
                        idx = len(temp) - 1
                    if st.session_state.app_config.use_stopwords:
                        for doc in temp.documents:
                            doc.text.remove_stopwords()
                    result = temp.compare(idx, model=load_model(), inplace=False)
                    st.session_state.app_rankings = []
                    st.session_state.ranking_pending = False
                    st.session_state.params_pending = False
                    for i, doc in enumerate(result.documents):
                        if doc.name != st.session_state.app_source.name:
                            st.session_state.app_rankings.append({
                                'doc': doc,
                                'score': result.similarities[i],
                                'rank': len(st.session_state.app_rankings) + 1
                            })
                st.rerun()
            
            if st.session_state.app_rankings:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="stat-box"><div class="stat-value">{len(st.session_state.app_rankings)}</div><div class="stat-label">Documents</div></div>', unsafe_allow_html=True)
                with c2:
                    avg = sum(r['score'] for r in st.session_state.app_rankings) / len(st.session_state.app_rankings)
                    st.markdown(f'<div class="stat-box"><div class="stat-value">{avg*100:.1f}%</div><div class="stat-label">Score moyen</div></div>', unsafe_allow_html=True)
                with c3:
                    top = st.session_state.app_rankings[0]['score']
                    st.markdown(f'<div class="stat-box"><div class="stat-value">{top*100:.1f}%</div><div class="stat-label">Score max</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="stat-box"><div class="stat-value">{len(st.session_state.app_alignments)}</div><div class="stat-label">Alignements</div></div>', unsafe_allow_html=True)
                
                st.divider()
                
                for r in st.session_state.app_rankings:
                    c1, c2, c3 = st.columns([1, 5, 2])
                    with c1:
                        color = "#d97706" if r['rank'] <= 3 else "#9ca3af"
                        st.markdown(f"<h2 style='color:{color};text-align:center;margin:0'>#{r['rank']}</h2>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"**{r['doc'].name}**")
                        st.caption(f"{len(r['doc'].text.words)} mots · {r['doc'].text.n_sentences} phrases")
                    with c3:
                        st.markdown(f"<h3 style='color:#2563eb;margin:0'>{r['score']*100:.1f}%</h3>", unsafe_allow_html=True)
                        st.progress(r['score'])
                    st.markdown("---")
                
                doc_names = [r['doc'].name for r in st.session_state.app_rankings]
                
                source_name = st.session_state.app_source.name
                st.markdown(f'Sélectionner un document à comparer avec : **"{source_name}"**')
                selected_doc = st.selectbox("", doc_names, key="select_align_doc", label_visibility="collapsed")
                
                if st.button("Aligner les textes →", use_container_width=True, type="primary"):
                    for r in st.session_state.app_rankings:
                        if r['doc'].name == selected_doc:
                            st.session_state.app_target = r['doc']
                            break
                    st.session_state.app_alignments = []
                    st.session_state.need_realign = True
                    st.session_state.active_tab = 2
                    st.rerun()
    
    # ===================== PAGE 3) RÉSULTATS =====================
    elif st.session_state.active_tab == 2:
        if not st.session_state.app_source:
            st.warning("⚠️ Chargez un texte source dans l'onglet Entrée")
        elif not st.session_state.app_target:
            st.info("ℹ️ Sélectionnez un document depuis l'onglet Classement")
        else:
            cfg = st.session_state.app_config
            
            th = st.session_state.align_th
            
            # Si params_pending, attendre que l'utilisateur relance manuellement
            # sauf si need_realign est True (premier calcul ou nouveau document)
            if st.session_state.params_pending and not st.session_state.need_realign and st.session_state.app_alignments:
                pass  # warning already shown in sidebar, wait for user action
            
            if st.session_state.need_realign:
                st.session_state.need_realign = False
                st.session_state.params_pending = False
                st.session_state.align_page = 0
                
                progress_bar = st.progress(0, text="Préparation...")
                start_time = time.time()
                
                progress_bar.progress(10, text="Chargement du modèle...")
                model = load_model()
                
                progress_bar.progress(20, text="Configuration...")
                run_cfg = ComparisonConfig(**st.session_state.app_config.to_dict())
                run_cfg.similarity_threshold = th
                
                progress_bar.progress(30, text="Segmentation en phrases...")
                
                text1 = st.session_state.app_source.text
                text2 = st.session_state.app_target.text
                text1.default()
                text2.default()
                
                if st.session_state.exclude_stopwords:
                    text1.remove_stopwords()
                    text2.remove_stopwords()
                
                comp = PairText(text1, text2, config=run_cfg)
                
                progress_bar.progress(50, text="Alignement par phrases...")
                
                st.session_state.app_alignments = comp.compare_n_grams(
                    model=model, score_threshold=th
                )
                st.session_state.app_comparateur = comp
                st.session_state.params_pending = False
                st.session_state.pending_params = set()
                
                elapsed = time.time() - start_time
                progress_bar.progress(100, text=f"✓ Terminé en {elapsed:.1f}s")
                time.sleep(0.3)
                progress_bar.empty()
                st.rerun()
            
            # Résultats
            if st.session_state.app_alignments:

                if st.session_state.app_comparateur:
                    # Stopwords pour filtrage mots communs côté JS
                    stopwords = Global_stuff.STOPWORDS

                    # ── Tri dans la page Résultats ──
                    sort_options = ["Lexicale (Jaccard)", "Lexicale (Overlap)", "Sémantique", "Combinée (60/40)"]
                    cur_sort = st.session_state.get('align_sort_mode', 'Combinée (60/40)')
                    sort_choice = st.radio(
                        "Trier les résultats par similarité",
                        sort_options,
                        index=sort_options.index(cur_sort) if cur_sort in sort_options else 3,
                        horizontal=True,
                        key="results_sort_mode",
                        help="**Lexicale (Jaccard)** : ratio mots en commun / union des mots (hors stopwords). Sensible à la longueur des segments.\n\n**Lexicale (Overlap)** : ratio mots en commun / plus petit des deux segments (hors stopwords). Corrige le biais de longueur de Jaccard.\n\n**Sémantique** : score cosinus entre les embeddings. Mesure la proximité de sens indépendamment de la forme lexicale.\n\n**Combinée (60/40)** : 60% sémantique + 20% Jaccard + 20% Overlap, après normalisation min-max."
                    )
                    if sort_choice != st.session_state.align_sort_mode:
                        st.session_state.align_sort_mode = sort_choice
                        st.session_state.align_page = 0
                        st.rerun()

                    mode_map = {
                        "Lexicale (Jaccard)": "lexical_jaccard",
                        "Lexicale (Overlap)": "lexical_overlap",
                        "Sémantique":         "semantic",
                        "Combinée (60/40)":   "combined",
                    }
                    sorted_alignments = st.session_state.app_comparateur.sort_alignments(
                        st.session_state.app_alignments,
                        mode=mode_map.get(sort_choice, "combined")
                    )

                    # ── Pagination ──
                    BLOCKS_PER_PAGE = 50
                    all_matches = sorted_alignments
                    total_blocks = len(all_matches)
                    total_pages = max(1, (total_blocks + BLOCKS_PER_PAGE - 1) // BLOCKS_PER_PAGE)
                    
                    # Clamp page
                    if st.session_state.align_page >= total_pages:
                        st.session_state.align_page = total_pages - 1
                    if st.session_state.align_page < 0:
                        st.session_state.align_page = 0
                    
                    page = st.session_state.align_page
                    start_idx = page * BLOCKS_PER_PAGE
                    end_idx = min(start_idx + BLOCKS_PER_PAGE, total_blocks)
                    page_matches_raw = all_matches[start_idx:end_idx]
                    
                    html = make_html(
                        page_matches_raw,
                        st.session_state.app_comparateur,
                        stopwords,
                        start_offset=start_idx,
                        total_alignments=total_blocks,
                        raw_scores=getattr(st.session_state.app_comparateur, '_raw_scores', {})
                    )
                    components.html(html, height=1650, scrolling=True)
                    
                    # Navigation si plus d'une page
                    if total_pages > 1:
                        _, nav_col, _ = st.columns([1, 4, 1])
                        with nav_col:
                            c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 1, 1])
                            with c1:
                                if st.button("⟨⟨ Début", disabled=(page == 0), key="pg_first", use_container_width=True):
                                    st.session_state.align_page = 0
                                    st.rerun()
                            with c2:
                                if st.button("⟨ Préc.", disabled=(page == 0), key="pg_prev", use_container_width=True):
                                    st.session_state.align_page -= 1
                                    st.rerun()
                            with c3:
                                st.markdown(
                                    f"<div style='text-align:center;padding:0.5rem;font-weight:600;color:#374151'>"
                                    f"Page {page+1} / {total_pages} — Blocs {start_idx+1}-{end_idx} sur {total_blocks}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            with c4:
                                if st.button("Suiv. ⟩", disabled=(page >= total_pages - 1), key="pg_next", use_container_width=True):
                                    st.session_state.align_page += 1
                                    st.rerun()
                            with c5:
                                if st.button("Fin ⟩⟩", disabled=(page >= total_pages - 1), key="pg_last", use_container_width=True):
                                    st.session_state.align_page = total_pages - 1
                                    st.rerun()
                

            else:
                st.info("Aucun alignement trouvé avec ces paramètres. Essayez de baisser le seuil.")
    
    # ===================== PAGE 4) STATS ET VIZ =====================
    elif st.session_state.active_tab == 4:
        if not hasattr(st.session_state, 'app_source') or st.session_state.app_source is None:
            st.warning("⚠️ Chargez un texte source dans l'onglet Entrée")
        elif not hasattr(st.session_state, 'app_alignments') or not st.session_state.app_alignments:
            st.info("ℹ️ Lancez un alignement dans l'onglet Résultats pour afficher les statistiques et visualisations.")
        else:
            stats1 = compute_text_stats(st.session_state.app_source.text)
            stats2 = compute_text_stats(st.session_state.app_target.text)

            # ── Visualisations ──
            st.markdown('<div style="font-size:1.1rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#111827;text-align:center;margin:1.5rem 0 1rem">📈 Visualisations</div>', unsafe_allow_html=True)

            viz_c1, viz_c2 = st.columns(2)
            with viz_c1:
                heatmap_html = render_heatmap(
                    st.session_state.app_alignments,
                    st.session_state.app_source.text,
                    st.session_state.app_target.text,
                    st.session_state.app_source.name,
                    st.session_state.app_target.name
                )
                components.html(heatmap_html, height=480)
            with viz_c2:
                radar_html = render_radar(
                    stats1, stats2,
                    st.session_state.app_source.name,
                    st.session_state.app_target.name,
                    Global_stuff.STOPWORDS
                )
                components.html(radar_html, height=480)

            sankey_html = render_sankey(
                st.session_state.app_alignments,
                st.session_state.app_source.text,
                st.session_state.app_target.text,
                st.session_state.app_source.name,
                st.session_state.app_target.name
            )
            components.html(sankey_html, height=400)

            # ── Timeline parallèle ──
            timeline_html = render_timeline(
                st.session_state.app_alignments,
                st.session_state.app_source.text,
                st.session_state.app_target.text,
                st.session_state.app_source.name,
                st.session_state.app_target.name
            )
            components.html(timeline_html, height=240)

            # ── Histogramme + Densité côte à côte ──
            viz_c3, viz_c4 = st.columns(2)
            with viz_c3:
                hist_html = render_score_histogram(st.session_state.app_alignments)
                components.html(hist_html, height=310)
            with viz_c4:
                density_html = render_density(
                    st.session_state.app_alignments,
                    st.session_state.app_source.text,
                    st.session_state.app_target.text,
                    st.session_state.app_source.name,
                    st.session_state.app_target.name
                )
                components.html(density_html, height=310)

            # ── Nuage de mots différentiel ──
            wordcloud_html = render_wordcloud(
                stats1, stats2,
                st.session_state.app_source.name,
                st.session_state.app_target.name,
                Global_stuff.STOPWORDS
            )
            components.html(wordcloud_html, height=300)

            # ── Matrice de co-occurrence ──
            cooc_html = render_cooccurrence(
                st.session_state.app_alignments,
                st.session_state.app_source.text,
                st.session_state.app_target.text,
                Global_stuff.STOPWORDS
            )
            components.html(cooc_html, height=520)

            # ── Statistiques ──
            st.divider()
            st.markdown('<div style="font-size:1.1rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#111827;text-align:center;margin:1.5rem 0 1rem">📊 Statistiques</div>', unsafe_allow_html=True)
            render_stats(
                stats1, stats2,
                st.session_state.app_source.name,
                st.session_state.app_target.name
            )

    # ===================== PAGE 5) GUIDE =====================
    elif st.session_state.active_tab == 3:
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
                font-size: 0.75rem; font-weight: 600; color: #fff;
                margin-right: 4px; vertical-align: middle;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("## 📖 Guide d'utilisation")
        st.caption("VERSUS — Outil de comparaison textuelle par similarité sémantique et lexicale")

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
  <text x="195" y="283" text-anchor="middle" font-size="10" fill="#6b7280">Embeddings all-MiniLM-L6-v2 · cosinus · tri décroissant</text>
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
            supprimé via le bouton ✕. Le filtre par mots-clés (🔎) permet de restreindre le corpus avant
            traitement — les fichiers non retenus restent visibles dans la zone de dépôt.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-section">
            <h3>2 — Traitement</h3>
            <p>Cliquez sur <b>Classer les textes cibles par similarité →</b> pour vectoriser et trier les
            documents par score de similarité globale. La pondération (<b>BM25</b> ou <b>TF-IDF</b>) est
            utilisée pour calculer le poids des embeddings de chaque phrase avant agrégation en vecteur
            de document.</p>
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
            (mots, phrases, diversité lexicale, fréquences communes top 10, mots propres top 10)
            et les visualisations : Heatmap, Radar, Sankey, Timeline, Histogramme des scores,
            Courbe de densité, Nuage de mots différentiel, Matrice de co-occurrence.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Pipeline de traitement")

        st.markdown("""
        <div class="guide-section">
            <p>
                <b>Passe 1 — Embeddings :</b> chaque phrase est encodée par <code>all-MiniLM-L6-v2</code>.
                Les paires sont comparées par cosinus (ou FAISS si le texte cible dépasse 50 phrases).
                C'est la passe principale — détecte les rapprochements sémantiques.<br><br>
                <b>Passe 2 — N-grams (optionnelle) :</b> passe lexicale sur les paires dont le score cosinus
                tombe dans la zone d'incertitude [0.60–0.85]. Récupère des correspondances que les embeddings
                auraient sous-scorées.<br><br>
                <b>Calcul des scores lexicaux :</b> pour chaque paire retenue, deux métriques sont calculées
                sur les mots hors stopwords —
                <b>Jaccard</b> (mots communs / union des deux ensembles, sensible à la longueur des segments)
                et <b>Overlap</b> (mots communs / plus petit segment, corrige ce biais de longueur).
                Les scores cosinus, Jaccard et Overlap sont normalisés min-max puis fusionnés selon
                <b>Option B : 60 % sémantique + 20 % Jaccard + 20 % Overlap</b> pour produire le score combiné.<br><br>
                <b>Passe 3 — SequenceMatcher :</b> calcul des suppressions et insertions caractère par caractère
                à l'intérieur de chaque paire alignée (affiché à la demande par bloc).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Paramètres")

        st.markdown("""
        <div class="guide-section">
            <p>Tous les paramètres sont accessibles dans la barre latérale. Les vignettes colorées
            sous le logo VERSUS affichent leur état en temps réel. Un avertissement apparaît sous
            chaque paramètre modifié si un alignement existe déjà.</p>
            <p>
                <b>Exclure les stopwords</b> — Agit en amont sur les embeddings (classement et alignement).
                Améliore la précision sur les textes à contenu dense.<br>
                <b>Méthode de pondération</b> — BM25 ou TF-IDF. Calcule le poids des embeddings de chaque
                phrase pour le vecteur de document (classement uniquement).<br>
                <b>Seuil cosinus</b> — Score minimum pour retenir une paire. Paramètre avec le plus d'impact
                sur le volume de résultats.<br>
                <b>Seuil adaptatif</b> — Favorise les passages longs par rapport aux courts lors du filtrage.<br>
                <b>Affinage intermédiaire</b> — Active la passe 2 (n-grams, lexical) sur la zone [0.60–0.85].<br>
                <b>Agrégation bidirectionnelle</b> — Détecte les correspondances asymétriques
                (cible → source). Double le temps de calcul.
            </p>
            <p>Les <b>stopwords</b> sont chargées depuis <code>stopwords.txt</code>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="guide-section">
            <h3>Export CSV</h3>
            <p>L'export des blocs sélectionnés inclut : numéro, positions source/cible, textes (200 car. max)
            et nombre de suppressions/insertions par bloc.</p>
        </div>
        """, unsafe_allow_html=True)


# =================================================================
#  Rendu HTML des alignements
#  Rendu HTML des alignements
# =================================================================

def make_html(matches, comp, stopwords, start_offset=0, total_alignments=None, raw_scores=None):
    ctx = 50
    
    COLOR_MATCH = "rgba(16,185,129,0.35)"
    BORDER_COLOR = "#10b981"

    import re as _re

    html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <style>
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:system-ui,sans-serif;background:#fafbfc;padding:1rem;font-size:13px;color:#111827}
        .m{background:#fff;border-radius:12px;margin-bottom:1rem;overflow:hidden;position:relative}
        .mh{padding:.75rem 1rem;font-weight:600;color:#374151;display:flex;justify-content:space-between;align-items:center;background:rgba(16,185,129,.12);border-bottom:1px solid #10b981}
        .mb{display:grid;grid-template-columns:1fr 1fr}
        .p{padding:1rem;border-right:1px solid #e5e7eb}
        .p:last-child{border-right:none}
        .ph{font-size:.7rem;text-transform:uppercase;color:#6b7280;margin-bottom:.5rem;font-weight:600}
        .pn{font-family:Georgia,serif;font-size:1rem;font-weight:600;margin-bottom:.75rem;color:#111827}
        .t{font-family:Georgia,serif;line-height:1.9;background:#f9fafb;padding:1rem;border-radius:6px;max-height:180px;overflow-y:auto;color:#111827}
        .sup{background:rgba(239,68,68,0.3);padding:1px 2px;border-radius:2px;font-weight:600}
        .ins{background:rgba(59,130,246,0.3);padding:1px 2px;border-radius:2px;font-weight:600}
        .cw{font-weight:900;text-decoration:underline;text-underline-offset:3px;font-size:1.15em;letter-spacing:0.03em;text-shadow:0 0 0.5px rgba(0,0,0,0.3);background:rgba(0,0,0,0.06);padding:0 2px;border-radius:2px}
        .ps{font-size:.7rem;color:#6b7280;margin-top:.5rem}
        .bt{padding:.4rem .75rem;background:#e5e7eb;border:none;border-radius:6px;cursor:pointer;font-size:.7rem;margin:0 2px;color:#374151}
        .bt:hover{background:#d1d5db}
        .cnt{font-size:.65rem;padding:3px 8px;border-radius:4px;font-weight:600;color:#374151;background:#f1f5f9;margin-left:.5rem}
        .ctrls{display:flex;align-items:center;gap:1rem;padding:.5rem 1rem;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:.75rem;flex-wrap:wrap}
        .ctrls label{display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none;color:#374151}
        .ctrls input[type=checkbox]{accent-color:#10b981;width:14px;height:14px}
        .lbl-cw{color:#059669;font-weight:600}
        .lbl-sel{color:#6b7280;font-weight:500}
        .sel-check{accent-color:#2563eb !important}
        .toolbar{display:flex;align-items:center;justify-content:space-between;padding:.75rem 1rem;background:#fff;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:1rem;position:relative}
        .toolbar-left{display:flex;align-items:center;gap:.75rem;font-size:.8rem;color:#374151}
        .toolbar-center{position:absolute;left:50%;transform:translateX(-50%);font-size:1.15rem;font-weight:700;color:#111827;white-space:nowrap}
        .export-btn{padding:.4rem .75rem;background:#f3f4f6;border:1px solid #d1d5db;border-radius:6px;cursor:pointer;font-size:.72rem;font-weight:600;color:#374151}
        .export-btn:hover{background:#e5e7eb}
        .export-btn:disabled{background:#f9fafb;color:#9ca3af;border-color:#e5e7eb;cursor:not-allowed}
        .sel-all-btn{padding:.4rem .75rem;background:#f3f4f6;border:1px solid #d1d5db;border-radius:6px;cursor:pointer;font-size:.72rem;color:#374151}
        .sel-all-btn:hover{background:#e5e7eb}
        .lex-bar-wrap{display:inline-flex;align-items:center;gap:5px;margin-left:.6rem;vertical-align:middle}
        .lex-bar-bg{width:48px;height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle;border:1px solid #9ca3af}
        .lex-bar-fg{height:100%;border-radius:3px}
        .lex-pct{font-size:.6rem;font-weight:600;min-width:26px;text-align:right}
        .score-bars{display:inline-flex;align-items:center;gap:8px;margin-left:.75rem;vertical-align:middle}
        .score-item{display:inline-flex;align-items:center;gap:4px}
        .score-lbl{font-size:.58rem;font-weight:600;text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}
        .score-bar-bg{width:44px;height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden;display:inline-block;border:1px solid #9ca3af}
        .score-bar-fg{height:100%;border-radius:3px;display:block}
        .score-pct{font-size:.6rem;font-weight:600;min-width:24px}
    </style></head><body>"""
    
    # Toolbar with select all + export
    html += f"""
    <div class="toolbar">
        <div class="toolbar-left">
            <button class="sel-all-btn" onclick="toggleAll()">☑ Tout sélectionner</button>
            <span id="sel-count">0 sélectionné(s)</span>
        </div>
        <div class="toolbar-center">{total_alignments} alignement(s) au total</div>
        <button class="export-btn" id="export-btn" onclick="exportCSV()">↓ Export CSV</button>
    </div>"""
    
    for i, match in enumerate(matches):
        pos1, pos2 = match[0], match[1]

        # Récupération des scores depuis _raw_scores
        rs = (raw_scores or {}).get((pos1, pos2), {})
        sem_raw  = rs.get('semantic',        0.0)
        jac_raw  = rs.get('lexical_jaccard', 0.0)
        ovl_raw  = rs.get('lexical_overlap', 0.0)
        comb_raw = rs.get('combined',        0.0)

        sem_pct  = int(round(sem_raw  * 100))
        jac_pct  = int(round(jac_raw  * 100))
        ovl_pct  = int(round(ovl_raw  * 100))
        comb_pct = int(round(comb_raw * 100))

        def lex_color(p):
            if p >= 40: return "#10b981", "#065f46"
            if p >= 15: return "#f59e0b", "#92400e"
            return "#ef4444", "#991b1b"

        jc, jpc = lex_color(jac_pct)
        oc, opc = lex_color(ovl_pct)

        def bar(label, pct, bar_col, pct_col, title):
            return (
                f'<span class="score-item" title="{title}">'
                f'<span class="score-lbl" style="color:{pct_col}">{label}</span>'
                f'<span class="score-bar-bg"><span class="score-bar-fg" style="width:{pct}%;background:{bar_col}"></span></span>'
                f'<span class="score-pct" style="color:{pct_col}">{pct}%</span>'
                f'</span>'
            )

        scores_badge = (
            f'<span class="score-bars">'
            + bar("jac.", jac_pct, "#628990", "#374151", "")
            + bar("ovl.", ovl_pct, "#628990", "#374151", "")
            + bar("sém.", sem_pct, "#628990", "#374151", "")
            + bar("cmb.", comb_pct,"#628990", "#374151", "")
            + f'</span>'
        )
        
        html += f"""
        <div class="m" style="border:2px solid {BORDER_COLOR}" id="block_{i}">
            <div class="mh">
                <span>#{start_offset+i+1}{scores_badge}</span>
                <div><button class="bt" onclick="c({i},-100)">−</button><button class="bt" onclick="r({i})">Reset</button><button class="bt" onclick="c({i},100)">+</button></div>
            </div>
            <div class="ctrls">
                <label><input type="checkbox" id="chk_sel_{i}" class="sel-check" onchange="updateSelCount()"><span class="lbl-sel">Sélectionner</span></label>
                <span style="color:#d1d5db">│</span>
                <label><input type="checkbox" id="chk_cw_{i}" onchange="u({i})"><span class="lbl-cw">Mots communs</span></label>
            </div>
            <div class="mb">
                <div class="p"><div class="ph">Texte Source</div><div class="pn">{comp.text1.name}</div><div class="t" id="a{i}"></div><div class="ps">Pos: {pos1[0]}-{pos1[1]}</div></div>
                <div class="p"><div class="ph">Texte Cible</div><div class="pn">{comp.text2.name}</div><div class="t" id="b{i}"></div><div class="ps">Pos: {pos2[0]}-{pos2[1]}</div></div>
            </div>
        </div>"""
    
    t1 = comp.text1.origin_content.replace('\\','\\\\').replace('`','\\`').replace('$','\\$').replace('\r',' ').replace('\n',' ')
    t2 = comp.text2.origin_content.replace('\\','\\\\').replace('`','\\`').replace('$','\\$').replace('\r',' ').replace('\n',' ')
    
    mjs = []
    for m_item in matches:
        mjs.append([list(m_item[0]), list(m_item[1])])
    
    # Escape stopwords for JS
    sw_escaped = json.dumps(sorted(list(stopwords)))
    
    html += f"""
    <script>
    const t1=`{t1}`,t2=`{t2}`,m={json.dumps(mjs)};
    const startOffset={start_offset};
    const BG='{COLOR_MATCH}';
    const stopWords=new Set({sw_escaped});
    let cs=new Array(m.length).fill({ctx});
    let allSelected=false;
    function e(t){{let d=document.createElement('div');d.textContent=t;return d.innerHTML}}
    
    // --- CORRECTION 3 : MOTS COMMUNS PAR BLOC (LOGIQUE JS AMÉLIORÉE) ---
    function getBlockCW(i){{
        let p1=m[i][0],p2=m[i][1];
        let context = cs[i];
        
        // On récupère TOUT le texte visible (Contexte + Match) pour identifier les mots communs
        let srcFull = t1.substring(Math.max(0, p1[0] - context), Math.min(t1.length, p1[1] + context)).toLowerCase();
        let tgtFull = t2.substring(Math.max(0, p2[0] - context), Math.min(t2.length, p2[1] + context)).toLowerCase();
        
        let w1 = new Set(srcFull.replace(/[.,;:!?'"()\\[\\]«»…–—‘’“”‹›°]/g,' ').split(/\\s+/).filter(w=>w.length>2&&!stopWords.has(w)));
        let w2 = new Set(tgtFull.replace(/[.,;:!?'"()\\[\\]«»…–—‘’“”‹›°]/g,' ').split(/\\s+/).filter(w=>w.length>2&&!stopWords.has(w)));
        
        let common = new Set();
        w1.forEach(w=>{{if(w2.has(w)) common.add(w)}});
        return common;
    }}
    
    function markCW(txt,cwSet){{
        if(!cwSet || cwSet.size === 0) return txt;
        return txt.replace(/\\S+/g, function(w){{
            let lw = w.toLowerCase().replace(/[.,;:!?'"()\\[\\]«»…–—‘’“”‹›°]/g,'');
            if(cwSet.has(lw)) return '<span class="cw">'+w+'</span>';
            return w;
        }});
    }}
    
    function hPanel(t,c,p,showCW,cwSet){{
        let s=p[0],x=p[1];
        let pr = e(t.substring(Math.max(0,s-c),s));
        let suf = e(t.substring(x,Math.min(t.length,x+c)));
        let seg = e(t.substring(s,x));
        if(showCW) seg=markCW(seg,cwSet);
        let inner='<span style="background:'+BG+';padding:2px 4px;border-radius:3px">'+seg+'</span>';
        if(showCW){{ pr=markCW(pr,cwSet); suf=markCW(suf,cwSet); }}
        return pr+inner+suf;
    }}
    
    function u(i){{
        let cw=document.getElementById('chk_cw_'+i).checked;
        let cwSet=cw?getBlockCW(i):new Set();
        document.getElementById('a'+i).innerHTML=hPanel(t1,cs[i],m[i][0],cw,cwSet);
        document.getElementById('b'+i).innerHTML=hPanel(t2,cs[i],m[i][1],cw,cwSet);
    }}
    
    function c(i,d){{cs[i]=Math.max({ctx},cs[i]+d);u(i)}}
    function r(i){{cs[i]={ctx};u(i)}}
    function updateSelCount(){{
        let n=0;
        for(let i=0;i<m.length;i++)if(document.getElementById('chk_sel_'+i).checked)n++;
        document.getElementById('sel-count').textContent=n+' sélectionné(s)';
    }}
    function toggleAll(){{
        allSelected=!allSelected;
        for(let i=0;i<m.length;i++)document.getElementById('chk_sel_'+i).checked=allSelected;
        updateSelCount();
    }}
    function exportCSV(){{
        let sel=[];
        for(let i=0;i<m.length;i++)if(document.getElementById('chk_sel_'+i).checked)sel.push(i);
        if(sel.length===0){{alert('Sélectionnez au moins un bloc à exporter.');return;}}
        let rows=[['num','pos_source','pos_cible','texte_source','texte_cible']];
        for(let k=0;k<sel.length;k++){{
            let i=sel[k];
            let p1=m[i][0],p2=m[i][1];
            let src=t1.substring(p1[0],p1[1]).substring(0,200);
            let tgt=t2.substring(p2[0],p2[1]).substring(0,200);
            rows.push([(startOffset+i+1),p1[0]+'-'+p1[1],p2[0]+'-'+p2[1],'"'+src.replace(/"/g,'""')+'"','"'+tgt.replace(/"/g,'""')+'"']);
        }}
        let csv=rows.map(r=>r.join(',')).join('\\n');
        let blob=new Blob([csv],{{type:'text/csv;charset=utf-8;'}});
        let url=URL.createObjectURL(blob);
        let a=document.createElement('a');a.href=url;a.download='alignements_selection.csv';a.click();
        URL.revokeObjectURL(url);
    }}
    for(let i=0;i<m.length;i++)u(i);
    </script></body></html>"""
    
    return html


if __name__ == "__main__":
    main()