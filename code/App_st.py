"""
Hyperparamètres : 
-N size des n_grams
-Stopwords
-Avec ou sans Stopwords
-Documents


Interactions : 
-Ajout de documents
-Ajout de corpus
-Filtre
-Compare
-Reset text

"""

import streamlit as st
import streamlit.components.v1 as components
from Corpus import Corpus
from Document import Document
from Comparateur import PairText
from typing import List, Tuple
from Global_stuff import Global_stuff
from sentence_transformers import SentenceTransformer
from time import sleep

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Versus",
    page_icon="VERSUSLOGO.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
if 'global_stuff' not in st.session_state :
    st.session_state.global_stuff = Global_stuff

@st.cache_resource
def load_model() : 
    return SentenceTransformer(st.session_state.global_stuff['MODEL_NAME'])


#Front-end complètement assisté par LLM parce que parlons franchement je savais pas comment on fait ça

def main():
    """Fonction principale de l'application Streamlit"""
    # Titre principal
    st.title("Versus")
        
    
    st.markdown("---")
    
    # Initialisation du state
    if 'corpus' not in st.session_state:
        st.session_state.corpus = Corpus()
    if 'filtered_corpus' not in st.session_state:
        st.session_state.filtered_corpus = None
    if 'comparateurs' not in st.session_state:
        st.session_state.comparateurs = dict()
    if 'comp_outputs' not in st.session_state:
        st.session_state.comp_outputs = dict()
    
    # Sidebar pour les paramètres globaux
    with st.sidebar:
        st.image("VERSUSLOGO.png", width=150)
        st.header("⚙️ Paramètres globaux")
        
        # Stopwords personnalisés
        st.subheader("Stopwords (Mots vides)")
        custom_stopwords = st.text_area(
            "Stopwords personnalisés (un par ligne)",
            value="\n".join(st.session_state.global_stuff.STOPWORDS),
            height=100
        )
        
        if st.button("Mettre à jour les stopwords"):
            st.session_state.global_stuff.STOPWORDS = set(
                word.strip().lower() for word in custom_stopwords.split('\n') 
                if word.strip()
            )
            st.success("Stopwords mis à jour!")

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["Gestion du Corpus", "Filtrage et Tri", "Comparaisons", "Guide"])
    
    with tab1:
        st.header("Gestion du Corpus")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Ajout de documents
            st.subheader("Ajouter des documents")
            
            # Upload de fichiers
            uploaded_files = st.file_uploader(
                "Choisir des fichiers .txt",
                type=['txt'],
                accept_multiple_files=True
            )
            
            if uploaded_files :
                if st.button("Ajouter les fichiers uploadés") :
                    for uploaded_file in uploaded_files :
                        doc = Document(name=uploaded_file.name, file_value=uploaded_file.getvalue())
                        res = st.session_state.corpus.add_doc(doc)
                        st.info(res)
                        sleep(0.5)
                    st.rerun()
            
            # Ajout manuel
            st.subheader("Ajouter un document manuellement")
            doc_name = st.text_input("Nom du document")
            doc_content = st.text_area("Contenu du document", height=150)
            
            if st.button("Ajouter le document") and doc_name and doc_content:
                doc = Document(name=doc_name, content=doc_content)
                res = st.session_state.corpus.add_doc(doc)
                st.info(res)
                sleep(0.5)
                st.rerun()
        
        with col2:
            # Informations sur le corpus
            st.subheader("Informations du corpus")
            st.metric("Nombre de documents", len(st.session_state.corpus))
            
            if len(st.session_state.corpus) > 0:
                total_words = sum(len(doc.text.words) for doc in st.session_state.corpus.documents)
                st.metric("Total de mots", total_words)
                total_sentences = sum(len(doc.text.sentences) for doc in st.session_state.corpus.documents)
                st.metric("Total de phrases", total_sentences)
        
        # Liste des documents
        if len(st.session_state.corpus) > 0:
            st.subheader("Documents dans le corpus")
            
            for i, doc in enumerate(st.session_state.corpus.documents):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{doc.name}** ({len(doc.text.words)} mots)")
                
                with col2:
                    if st.button("Voir", key=f"view_{i}"):
                        st.text_area(f"Contenu de {doc.name}", doc.text.origin_content[:500] + "..." if len(doc.text.origin_content) > 500 else doc.text.origin_content, height=100, key=f"content_{i}")
                
                with col3:
                    if st.button("Supprimer", key=f"delete_{i}"):
                        st.session_state.corpus = st.session_state.corpus - i
                        st.rerun()
    
    with tab2:
        st.header("Filtrage et tri du Corpus")
        
        if len(st.session_state.corpus) == 0:
            st.warning("Aucun document dans le corpus. Ajoutez des documents d'abord.")
        else:
            st.subheader("1. Filtrage")
            regex_pattern = st.text_input("Mots filtrant", placeholder="Exemple: 'HUGO'")
            
            if st.button("Appliquer le filtre") and regex_pattern:
                st.session_state.filtered_corpus = st.session_state.corpus.filter(regex_pattern)
                st.success(f"Filtre appliqué! {len(st.session_state.filtered_corpus)} document(s) correspondent. Corpus actuel mis à jour")
            
            if st.button("Réinitialiser le filtre"):
                st.session_state.filtered_corpus = None
                st.success("Filtre réinitialisé.")
            
            # Affichage du corpus actuel (filtré s'il existe, sinon l'original)
            current_corpus = st.session_state.filtered_corpus if st.session_state.filtered_corpus else st.session_state.corpus
            
            if len(current_corpus) > 0:
                st.subheader(f"Corpus actuel ({len(current_corpus)} documents)")
                for doc in current_corpus.documents:
                    st.write(f"- {doc.name}")

            # Sélection du document source pour le tri
            st.subheader("2. Tri")
            source_doc_name = st.selectbox(
                "Document source",
                current_corpus.get_documents_names(),
                key="source_doc_tri"
            )
            
            if source_doc_name:
                source_doc = current_corpus.get_doc_by_name(source_doc_name)
                
                # Option pour trier par similarité
                if st.button("Trier le corpus par similarité"):
                    index_doc = current_corpus.index(source_doc)
                    with st.spinner(text="Tri du corpus...", show_time=True) : 
                        sorted_corpus = current_corpus.compare(index_doc, model=load_model())

                    st.success("Corpus actuel trié par similarité!")

                    with st.container():
                        for i, doc in enumerate(sorted_corpus.documents):
                            st.write(f"{i+1}. {doc.name}")

                    #On actualise le corpus actuel : 
                    if st.session_state.filtered_corpus : 
                        st.session_state.filtered_corpus = sorted_corpus #Le corpus filtré trié
                        st.session_state.corpus = sorted_corpus + st.session_state.corpus #Le corpus filtré trié et les documents restants en bout de file
                    else  :
                        st.session_state.corpus = sorted_corpus 
                    
    
    with tab3:
        st.header("Comparaisons de Documents")
        
        current_corpus = st.session_state.filtered_corpus if st.session_state.filtered_corpus else st.session_state.corpus
        
        if len(current_corpus) < 2:
            st.warning("Il faut au moins 2 documents pour faire des comparaisons.")
        else:
            # Sélection du document source
            st.subheader("1. Sélectionner le document source")
            source_doc_name = st.selectbox(
                "Document source",
                current_corpus.get_documents_names(),
                key="source_doc_compare"
            )
            
            if source_doc_name:
                source_doc = current_corpus.get_doc_by_name(source_doc_name)
    
                # Sélection du document cible
                st.subheader("2. Sélectionner le document cible")
                available_targets = [name for name in current_corpus.get_documents_names() if name != source_doc_name]
                
                target_doc_name = st.selectbox(
                    "Document cible",
                    available_targets,
                    key="target_doc"
                )
                
                if target_doc_name:
                    target_doc = current_corpus.get_doc_by_name(target_doc_name)
                    
                    # Créer un comparateur
                    comparison_key_neutral = f"{source_doc_name}_vs_{target_doc_name}"

                    i = 1
                    comparison_key = comparison_key_neutral + "_" + str(i)
                    while comparison_key in st.session_state.comparateurs : 
                        i+=1
                        comparison_key = comparison_key_neutral + "_" + str(i)
                        

                    if st.button("Créer une comparaison"):
                        st.session_state.comparateurs[comparison_key] = PairText(source_doc.text, target_doc.text)
                        st.session_state.comp_outputs[comparison_key] = {'matches' : None, 'params' : None, 'html_output' : "Le résultat s'affichera ici."}
                        st.success(f"Comparaison créée: {source_doc_name} vs {target_doc_name}")
                        st.rerun()
        
        # Affichage des comparaisons existantes
        if st.session_state.comparateurs:
            st.subheader("3. Comparaisons actives")
            
            # Onglets pour chaque comparaison
            comparison_tabs = st.tabs(list(st.session_state.comparateurs.keys()))
            
            for i, (comp_key, comparateur) in enumerate(st.session_state.comparateurs.items()):
                with comparison_tabs[i]:
                    st.subheader(f"Comparaison: {comp_key}")
                    
                    # Paramètres de comparaison
                    size_choice, threshold_choice, stopword_checkbox, diff_checkbox, delete_comp  = st.columns(5)
                    
                    with size_choice:
                        n_size = st.slider("Taille des n-grams", 1, 10, 3, key=f"n_{comp_key}")
                    
                    with threshold_choice:
                        threshold = st.slider("Seuil de similarité", 0.8, 1.0, 0.93, 0.01, key=f"threshold_{comp_key}")

                    with stopword_checkbox : 
                        ignore_stopwords = st.checkbox("Supprimer les stopwords", key=f"stopwords_{comp_key}") 

                    with diff_checkbox:
                        show_diff = st.checkbox("Montrer les différences", key=f"diff_{comp_key}")

                        #On actualise dynamiquement l'affichage pour montrer les différences
                        if st.session_state.comp_outputs[comp_key]['matches'] is not None : 
                            matches = st.session_state.comp_outputs[comp_key]['matches']
                            st.session_state.comp_outputs[comp_key]['html_output'] = create_navigation_interface(matches, comparateur, comp_key, show_diff)
                    
                    with delete_comp:
                        if st.button("Supprimer", key=f"remove_{comp_key}"):
                            del st.session_state.comparateurs[comp_key]
                            del st.session_state.comp_outputs[comp_key]
                            st.rerun()

                    if st.button("Comparer", key=f"compare_{comp_key}"):
                        # Effectuer la comparaison
                        
                        if ignore_stopwords : 
                            with st.spinner("Suppression des mots vides", show_time=True) :
                                comparateur.remove_stopwords_texts()

                        else : 
                            comparateur.set_default_texts()
                        
                        if threshold > 0.99 : 
                            threshold = 0.99
                        
                        current_params = (n_size, threshold, ignore_stopwords)

                        #Si la comp_key n'est pas là ou bien si les paramètres ont changé, on relance la comparaison
                        if st.session_state.comp_outputs[comp_key]['params'] != current_params  : 
                            st.session_state.comp_outputs[comp_key]['params'] = current_params
                            with st.spinner("Comparaison en cours...", show_time=True) :
                                st.session_state.comp_outputs[comp_key]['matches'] = comparateur.compare_n_grams(n=n_size, model=load_model(), score_threshold=threshold)

                        matches = st.session_state.comp_outputs[comp_key]['matches']

                        navigation_html = create_navigation_interface(matches, comparateur, comp_key, show_diff)
                        st.session_state.comp_outputs[comp_key]['html_output'] = navigation_html

                        with open("last_output.html", 'w', encoding="utf-8") as f :
                            f.write(navigation_html)
                    
                    components.html(st.session_state.comp_outputs[comp_key]['html_output'], height=1000, scrolling=True)

    with tab4 : 
        st.write("GUIDE : ")
        st.write("Pour éviter tout bug : ne pas mettre de titre contenant : des caractères spéciaux, des guillemets, des apostrophes")


def create_navigation_interface(matches, comparateur, comp_key, show_diff):
    """
    Crée une interface HTML pour naviguer et comparer les passages similaires entre deux textes.
    
    Args:
        matches: Liste de 4-uplets (pos1, pos2, diff1, diff2)
        comparateur: Objet avec text1 et text2
        comp_key: Clé unique pour cette comparaison
        show_diff: Booléen pour afficher les différences
    
    Returns:
        str: Code HTML de l'interface de navigation
    """
    
    context_size = st.session_state.global_stuff.INITIAL_CONTEXT_SIZE
    delta_size = st.session_state.global_stuff.DELTA_CONTEXT_SIZE
    


    # Début du HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comparaison de textes - {comp_key}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                color: #e0e0e0;
                line-height: 1.6;
            }}
            
            .header {{
                background: #2d2d2d;
                color: #f0f0f0;
                padding: 20px;
                border-radius: 4px;
                margin-bottom: 20px;
                text-align: center;
                border: 1px solid #404040;
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 24px;
                color: #ffffff;
            }}
            
            .stats {{
                background: #2d2d2d;
                color: #cccccc;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 20px;
                border: 1px solid #404040;
                text-align: center;
            }}
            
            .match-container {{
                background: #2d2d2d;
                border-radius: 4px;
                margin-bottom: 20px;
                border: 1px solid #404040;
                overflow: hidden;
                transition: all 0.2s ease;
            }}
            
            .match-container:hover {{
                border-color: #555555;
                transform: translateY(-1px);
            }}
            
            .match-header {{
                background: #404040;
                color: #f0f0f0;
                padding: 15px 20px;
                font-weight: bold;
                cursor: pointer;
                transition: background 0.2s ease;
                border-bottom: 1px solid #555555;
            }}
            
            .match-header:hover {{
                background: #4a4a4a;
            }}
            
            .match-content {{
                display: flex;
                min-height: 200px;
            }}
            
            .text-panel {{
                flex: 1;
                padding: 20px;
                border-right: 1px solid #404040;
                position: relative;
                background: #252525;
            }}
            
            .text-panel:last-child {{
                border-right: none;
            }}
            
            .text-panel h3 {{
                margin: 0 0 15px 0;
                color: #cccccc;
                font-size: 16px;
                padding-bottom: 10px;
                border-bottom: 1px solid #555555;
            }}
            
            .text-content {{
                background: #1e1e1e;
                color: #e0e0e0;
                padding: 15px;
                border-radius: 3px;
                font-family: 'Times New Roman';
                font-size: 22px;
                line-height: 1.2em;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #404040;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            .text-content:hover {{
                background: #262626;
                border-color: #666666;
            }}
            
            .text-content.expanded {{
                max-height: none;
                background: #1a1a1a;
                border-color: #777777;
            }}
            
            .diff-color {{
                color: {st.session_state.global_stuff.COLORS['diff']};
                padding: 1px 2px;
                border-radius: 2px;
                font-weight: bold;
            }}
            
            .match-highlight {{
                background-color: #5B9E56;
                padding: 1px 2px;
                border-radius: 1px;
            }}
            
            .context-controls {{
                margin: 10px 0;
                text-align: center;
            }}
            
            .context-btn {{
                background: #404040;
                color: #e0e0e0;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin: 0 3px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s ease;
            }}
            
            .context-btn:hover {{
                background: #4a4a4a;
                border-color: #666666;
            }}
            
            .context-btn:disabled {{
                background: #333333;
                color: #666666;
                cursor: not-allowed;
                border-color: #333333;
            }}
            
            .navigation {{
                position: sticky;
                top: 20px;
                background: #2d2d2d;
                padding: 15px;
                border-radius: 4px;
                border: 1px solid #404040;
                margin-bottom: 20px;
                text-align: center;
            }}
            
            .nav-btn {{
                background: #404040;
                color: #e0e0e0;
                border: 1px solid #555555;
                padding: 10px 20px;
                margin: 0 5px;
                border-radius: 3px;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            .nav-btn:hover {{
                background: #4a4a4a;
                border-color: #666666;
            }}
            
            .scrollbar-custom::-webkit-scrollbar {{
                width: 8px;
            }}
            
            .scrollbar-custom::-webkit-scrollbar-track {{
                background: #2d2d2d;
                border-radius: 2px;
            }}
            
            .scrollbar-custom::-webkit-scrollbar-thumb {{
                background: #555555;
                border-radius: 2px;
            }}
            
            .scrollbar-custom::-webkit-scrollbar-thumb:hover {{
                background: #666666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Comparaison de Textes</h1>
            <p>{comp_key}</p>
        </div>
        
        <div class="stats">
            <strong>{len(matches)} correspondance(s) trouvée(s)</strong>
            {' | Différences mises en évidence' if show_diff else ''}
        </div>
        
        <div class="navigation">
            <button class="nav-btn" onclick="expandAll()">Tout Étendre</button>
            <button class="nav-btn" onclick="resetAll()">Tout Reset</button>
            <button class="nav-btn" onclick="collapseAll()">Tout Réduire</button>
        </div>
    """
    
    # Générer chaque correspondance
    for i, match in enumerate(matches):
        # Gérer le cas où matches peut être des 2-uplets ou 4-uplets
        if len(match) == 2:
            pos1, pos2 = match
            diff1, diff2 = None, None
        else:
            pos1, pos2, diff1, diff2 = match
        
        # Extraire les textes correspondants
        # Calculer un score de similarité approximatif
        
        html += f"""
        <div class="match-container" id="match-{i}">
            <div class="match-header" onclick="toggleMatch({i})">
                Correspondance #{i+1}
                <div class="context-controls">
                    <button class="context-btn" onclick="changeContext({i}, -{delta_size})">Réduire le contexte</button>
                    <button class="context-btn" onclick="resetContext({i})">Reset le contexte</button>
                    <button class="context-btn" onclick="changeContext({i}, {delta_size})">Aggrandir le contexte</button>
                </div>
            </div>
            
            <div class="match-content" id="content-{i}">

                <div class="text-panel">
                    <h3>{comparateur.text1.name} (Position: {pos1[0]}-{pos1[1]})</h3>
                    <div class="text-content scrollbar-custom" id="text1-{i}" onclick="toggleExpand('text1-{i}')">
                    </div>
                </div>
                
                <div class="text-panel">
                    <h3>{comparateur.text2.name} (Position: {pos2[0]}-{pos2[1]})</h3>
                    <div class="text-content scrollbar-custom" id="text2-{i}" onclick="toggleExpand('text2-{i}')">
                    </div>
                </div>
            </div>
        </div>
        """
    
    # JavaScript pour l'interactivité
    html += f"""
    </body>
        <script>
// Données des textes pour le contexte dynamique
const text1 = `{repr(comparateur.text1)}`;
const text2 = `{repr(comparateur.text2)}`;
const matches = {str(matches).replace('(', '[').replace(')', ']')}; //Jeu dangereux : la ressemblance de syntaxe entre python et javascript : on bricole
const showDiff = {str(show_diff).lower()};
// Stockage des contextes actuels
let contexts_size = [];

// Initialiser les contextes
for (let i = 0; i < matches.length; i++) {{
    contexts_size[i] = {context_size}
}}

function escapeHtml(text) {{
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}}

function highlightTextWithDifferences(text, context_size, position, diffCoords) {{
    let start = position[0];
    let end = position[1];
    
    let prefix = text.substring(start-context_size,start)

    let inner = '<span class="match-highlight">';
    
    if (showDiff && diffCoords && diffCoords.length !== 0) {{
        // Trier les coordonnées
        diffCoords.sort((a, b) => a[0] - b[0]);
    
        let lastEnd = start;
        for (const [start_diff, end_diff] of diffCoords) {{
            inner += escapeHtml(text.substring(lastEnd, start_diff));
            
            const diffText = escapeHtml(text.substring(start_diff, end_diff));
            inner += `<span class="diff-color">${{diffText}}</span>`;
            
            lastEnd = end_diff;
        }}

        inner += escapeHtml(text.substring(lastEnd, end));
    }}
    else {{
        inner += escapeHtml(text.substring(start,end))
    }}

    inner += '</span>';
    let suffix = text.substring(end,end+context_size)
    let total = prefix + inner + suffix
    return total
}}

function updateTextContent(matchIndex) {{
    const match = matches[matchIndex];
    let [pos1, pos2, diff1, diff2] = [match[0], match[1], match[2], match[3]]

    let element = document.getElementById(`text1-${{matchIndex}}`);
    element.innerHTML = highlightTextWithDifferences(text1, contexts_size[matchIndex], pos1, diff1);
    
    element = document.getElementById(`text2-${{matchIndex}}`);
    element.innerHTML = highlightTextWithDifferences(text2, contexts_size[matchIndex], pos2, diff2);
}}

//On met immédiatement les premiers 
for (let i = 0; i<matches.length; i++){{
    updateTextContent(i);
}}

function changeContext(matchIndex, delta) {{
    contexts_size[matchIndex] = Math.max({context_size}, contexts_size[matchIndex] += delta);
    updateTextContent(matchIndex);
}}

function resetContext(matchIndex) {{
    contexts_size[matchIndex] = {context_size};
    updateTextContent(matchIndex);
}}

function collapseAll() {{
    for (let i = 0; i<matches.length; i++) {{
        changeContext(i, -default_delta)
    }}  
}}

function expandAll() {{
    for (let i = 0; i<matches.length; i++) {{
        changeContext(i, default_delta)
    }}
}}

function resetAll() {{
    for (let i = 0; i<matches.length; i++) {{
        resetContext(i)
    }}
}}

        </script>
    </html>
    """
    return html

main()
