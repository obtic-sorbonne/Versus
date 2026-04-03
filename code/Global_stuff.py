from platform import system
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('stopwords')

import re
#import torch

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_device(DEVICE)
#print(DEVICE)
class Global_stuff :
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    """
    if torch.cuda.is_available() :
        MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 
    else : 
        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
        """
    LANGUAGE = "french"
    DEPTH = -1
    MIN_SENT_LENGTH = 8
    ACCEPTED_EXTENSIONS = ["txt"]
    STOPWORDS_TOGGLED = False
    try :
        STOPWORDS = set([word.strip() for word in open("stopwords.txt", 'r', encoding='utf-8').readlines()])
    except FileNotFoundError : 
        STOPWORDS = nltk.corpus.stopwords.words(LANGUAGE)

    COLORS = {
        'sim' : "#3BC531",
        'diff' : '#ff0000'}
    
    INITIAL_CONTEXT_SIZE = 50
    DELTA_CONTEXT_SIZE = 100

    

#Hyperparamètres :
#les documents 
