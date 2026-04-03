"""
Constantes globales et configuration par défaut.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('stopwords', quiet=True)
import re


class Global_stuff:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LANGUAGE = "french"
    DEPTH = -1
    MIN_SENT_LENGTH = 8
    ACCEPTED_EXTENSIONS = ["txt", "docx", "pdf"]
    STOPWORDS_TOGGLED = False  # Contrôle si les stopwords sont actifs
    
    try:
        STOPWORDS = set([word.strip() for word in open("stopwords.txt", 'r', encoding='utf-8').readlines()])
    except FileNotFoundError:
        STOPWORDS = set(nltk.corpus.stopwords.words(LANGUAGE))

    COLORS = {
        'sim': "#3BC531",
        'diff': '#ff0000',
        'exact': 'rgba(16, 185, 129, 0.4)',
        'overlap': 'rgba(59, 130, 246, 0.35)',
        'reformulation': 'rgba(245, 158, 11, 0.4)',
        'semantic': 'rgba(147, 51, 234, 0.3)'
    }
    
    INITIAL_CONTEXT_SIZE = 50
    DELTA_CONTEXT_SIZE = 100
