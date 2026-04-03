"""
Export CSV avec métadonnées complètes pour reproductibilité.
"""

import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple
from Config import ComparisonConfig, get_document_hash


class ComparisonExporter:
    """Export CSV des résultats."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.export_date = datetime.now().isoformat()
    
    def export_corpus_ranking(self, source_doc_name: str, source_doc_hash: str,
                              ranked_docs: List[Dict[str, Any]], 
                              similarities: List[float],
                              filepath: str) -> str:
        """Exporte le classement du corpus."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            
            writer.writerow([
                "Rang", "Document", "Score_Similarite", 
                "Nb_Mots", "Nb_Phrases", "Hash_Document"
            ])
            
            for i, (doc_info, similarity) in enumerate(zip(ranked_docs, similarities)):
                writer.writerow([
                    i + 1,
                    doc_info.get("name", ""),
                    f"{similarity:.4f}",
                    doc_info.get("num_words", 0),
                    doc_info.get("num_sentences", 0),
                    doc_info.get("hash", "N/A")
                ])
            
            # Métadonnées
            writer.writerow([])
            writer.writerow(["--- METADONNEES ---"])
            writer.writerow(["Document_Source", source_doc_name])
            writer.writerow(["Hash_Source", source_doc_hash])
            writer.writerow(["Date_Export", self.export_date])
            writer.writerow(["Modele", self.config.model_name])
            writer.writerow(["Methode_Scoring", self.config.scoring_method])
            writer.writerow(["Strategie_Ponderation", self.config.weighting_strategy])
            writer.writerow(["Seuil_Similarite", self.config.similarity_threshold])
            writer.writerow(["Seuil_Adaptatif", "Oui" if self.config.adaptive_threshold else "Non"])
            writer.writerow(["Profil", self.config.profile_name])
            writer.writerow(["Config_Hash", self.config.get_hash()])
        
        return filepath
    
    def export_alignments(self, text1_name: str, text1_hash: str,
                         text2_name: str, text2_hash: str,
                         alignments: List[Tuple],
                         text1_content: str, text2_content: str,
                         filepath: str) -> str:
        """Exporte les alignements texte-texte."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            
            writer.writerow([
                "ID", "Position_Texte1", "Extrait_Texte1",
                "Position_Texte2", "Extrait_Texte2",
                "Longueur_Segment", "Seuil_Utilise"
            ])
            
            for i, alignment in enumerate(alignments):
                pos1, pos2 = alignment[0], alignment[1]
                
                seg1 = text1_content[pos1[0]:pos1[1]].replace('\n', ' ').replace(';', ',')[:300]
                seg2 = text2_content[pos2[0]:pos2[1]].replace('\n', ' ').replace(';', ',')[:300]
                
                segment_length = pos1[1] - pos1[0]
                threshold = self.config.get_adaptive_threshold(segment_length) if self.config.adaptive_threshold else self.config.similarity_threshold
                
                writer.writerow([
                    i + 1,
                    f"{pos1[0]}-{pos1[1]}",
                    seg1 + ("..." if len(seg1) >= 300 else ""),
                    f"{pos2[0]}-{pos2[1]}",
                    seg2 + ("..." if len(seg2) >= 300 else ""),
                    segment_length,
                    f"{threshold:.3f}"
                ])
            
            # Métadonnées
            writer.writerow([])
            writer.writerow(["--- METADONNEES ---"])
            writer.writerow(["Texte_1", text1_name])
            writer.writerow(["Hash_Texte_1", text1_hash])
            writer.writerow(["Texte_2", text2_name])
            writer.writerow(["Hash_Texte_2", text2_hash])
            writer.writerow(["Date_Export", self.export_date])
            writer.writerow(["Modele", self.config.model_name])
            writer.writerow(["Taille_N-grams", self.config.ngram_size])
            writer.writerow(["Seuil_Base", self.config.similarity_threshold])
            writer.writerow(["Seuil_Adaptatif", "Oui" if self.config.adaptive_threshold else "Non"])
            writer.writerow(["Coefficient_Alpha", self.config.adaptive_alpha])
            writer.writerow(["Methode_Scoring", self.config.scoring_method])
            writer.writerow(["Strategie_Ponderation", self.config.weighting_strategy])
            writer.writerow(["Profil", self.config.profile_name])
            writer.writerow(["Nb_Correspondances", len(alignments)])
            writer.writerow(["Config_Hash", self.config.get_hash()])
        
        return filepath
