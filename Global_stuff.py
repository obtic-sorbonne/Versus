"""
Constantes globales et configuration par défaut.
"""

import nltk
nltk.download('stopwords', quiet=True)
import re


class Global_stuff:
    #MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    LANGUAGE = "french"
    DEPTH = -1
    MIN_SENT_LENGTH = 8
    ACCEPTED_EXTENSIONS = ["txt", "docx", "pdf"]
    STOPWORDS_TOGGLED = False

    try:
        STOPWORDS = set([word.strip() for word in open("stopwords.txt", 'r', encoding='utf-8').readlines()])
    except FileNotFoundError:
        STOPWORDS = set(nltk.corpus.stopwords.words(LANGUAGE))
