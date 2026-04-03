from Global_stuff import *

from Chunks import Sentence, Word
from math import ceil
import numpy as np
from re import search
from itertools import accumulate

patterns = ""

class Text : 
    """A l'échelle du texte, et des modifications textuelles."""

    def __init__(self, name : str, content : str) : 
        """Attributs importants :
        name : name du texte. 
        origin_content : texte du document qui sera affiché. 
        content : contenu du document sur lequel les calculs seront effectués.
        sentences : liste des instances Sentence qui composent le texte."""

        self.name = name
        self.origin_content = content.strip().replace("\r", "\n")
        
        #On construit une liste de mot qui gardent en mémoire leur place dans le texte
        matches = list(re.finditer(r'\b\w+\b', self.origin_content))
        self.words = [Word(match.start(), match.end(), match.group()) for match in matches]
        self.vectorized = None
        self.default()

    def __getitem__(self, c) : 
        if isinstance(c, tuple) : 
            a,b = c[0], c[1]
            return self.origin_content.replace("\n", " ")[a:b]
        raise TypeError("Please get with a couple")

    def __repr__(self):
        return self.origin_content

    def default(self) : 
        """Réinitialise le contenu."""
        self.content = self.origin_content
        matches = list(re.finditer(r'\b\w+\b', self.origin_content))
        self.words = [Word(match.start(), match.end(), match.group()) for match in matches]
        self.update_sentences()
        return self

    def n_grams(self, n=1) :    
        #On renvoie des sous liste de taille n de mots  
        if n == 1 : 
            return self.words   
        return [self.words[i:i+n] for i in range(len(self.words)-n+1)]

    def remove_stopwords(self) :
        print('removing stopwords for ', self.name) 
        stopwords = Global_stuff.STOPWORDS
        word_contents = [word.content for word in self.words]
        print("done content")
        temp = [i for i,w in enumerate(word_contents) if w not in stopwords]
        print("done filter")
        self.words = list(np.array(self.words)[temp])
        print("done recup")
        #On actualise la liste de mots pour ne garder que les bons
        #self.words = [word for word in self.words if word.content not in stopwords]
        self.content = (" ".join([w.content for w in self.words])).strip()
        print("done dada")
        print('stopwords removed for ', self.name)
        return self

    def vectorize(self, model=SentenceTransformer(Global_stuff.MODEL_NAME)) : 
        if self.vectorized is None : 
            sentences_content = [s.content for s in self.sentences]
            self.vectorized = model.encode(sentences_content)
        
            print(f"Texte {self.name} entièrement traité et enregistré.")
        
        return self.vectorized
    
    def vectorize_window(self, model=SentenceTransformer(Global_stuff.MODEL_NAME), w=1) : 
        """Renvoie la matrice des embeddings des phrases dans la fenêtre de taille w, accompagnée des phrases"""
        if w == 1 : 
            return self.vectorize(), [[s] for s in self.sentences]
        elif w > 1 :
            step = ceil(w/2)
            #On construit les fenetres
            sentences_windows = [self.sentences[i:i+w] for i in range(0, self.n_sentences, step)]
            #On extrait le contenu
            sentences_windows_content = [" ".join([str(s) for s in window]) for window in sentences_windows]
            #On encode le tout
            windows_matrix = model.encode(sentences_windows_content)
            
            print(f"Texte {self.name} entièrement traité pour des séquences de {w} phrases")
            
            #On prend la première phrase de chaque fenetre
            return windows_matrix, sentences_windows
    
    def find_words(self, regex) :
        matches = re.finditer(regex, self.origin_content)
        return [(m.start(), m.end()) for m in matches]        

    def update_sentences(self) : 
        """Update la liste self.sentences pour correspondre à self.content."""
        sentences = split_sentences(self.content)
        
        to_delete = set()

        #On rattache les phrases trop courtes et on supprime les phrases sans lettres. 
        for i, e in enumerate(sentences) : 
            if search('[a-zA-Z]', e) is None : 
                to_delete.add(i)

            if i not in to_delete and len(e) < Global_stuff.MIN_SENT_LENGTH and i != len(sentences)-1:
                to_delete.add(i)
                sentences[i+1] = e + sentences[i+1] #On colle la phrase à la suivante si on la juge trop courte            

        for i in sorted(to_delete, reverse=True):
            del sentences[i] 

        indexes = [0] + list(accumulate([len(s) for s in sentences]))[:-1]

        self.sentences = [Sentence(i,s) for i,s in zip(indexes, sentences)]
        self.n_sentences = len(self.sentences)
        return self

def split_sentences(text):
    """On remplace tous les points de non-fin de phrases par des ¤, puis on les ramène aux points après avoir split."""
    def mask(text):
        substitutions = [
            #Dialogues sans guillemets 
            (r'—([^—]*?)[.;!?](.*?)\n', r'—\1¤\2\n'),
            # Points entre guillemets
            (r'“([^”]*?)[.;!?](.*?)”', r'“\1¤\2”'), (r'"([^"]*?)[.;!?](.*?)"', r'"\1¤\2"'), (r'«([^»]*?)[.;!?](.*?)»', r'«\1¤\2»'), (r"'([^']*?)[.;!?](.*?)'", r"'\1¤\2'"),
            # Parenthèses, accolades, crochets
            (r'\(([^\)]*?)[.;!?](.*?)\)', r'\(\1¤\2\)'), (r'\{([^\}]*?)[.;!?](.*?)\}', r'\{\1¤\2\}'), (r'\[([^\]]*?)[.;!?](.*?)\]', r'\[\1¤\2\]'),
            # Nombres à virgule
            (r'\b\d+\.\d+\b', lambda m: m.group(0).replace('.', '¤')),
            # Emails
            (r'[\w\.-]+@[\w\.-]+\.\w+', lambda m: m.group(0).replace('.', '¤')),
            # URLs
            (r'(?i)(https?:\/\/|w{3}\.)[\w.%\/?=&#:;+-]{5,}', lambda m: m.group(0).replace('.', '¤')),
            # Abréviations courantes
            (r'(?i)\b(etc|cf|ex|dr|mr|mrs|mlle|mme|ms|st|mt|vs|vol|chap|fig|al|ibid|op\.cit|loc\.cit|n\.b|e\.g|i\.e|ca)\.', lambda m: m.group(0).replace('.', '¤')), (r'\b(M)\.', lambda m: m.group(0).replace('.', '¤')),      
            # Abréviations scientifiques/techniques
            (r'(?i)\b(prof|univ|coll|col|vol|réf|ref|tabl|graph|edit|nat|reg|suppl|tom|act|gen|lib|acad|comm)\.(\d+)?', lambda m: m.group(0).replace('.', '¤')),
            # Lettres arabes + point
            (r'\b([\u0621-\u064A])\.', r'\1¤'),
            # Romains
            (r'(?i)\b([IVXLCDM]+)\.', r'\1¤'),
            # Lettres isolées ou initiales
            (r'\b[a-zA-Z]\.', r'\g<0>'.replace('.', '¤')),
            # Numérotation de début de ligne
            (r'^\d+\.', lambda m: m.group(0).replace('.', '¤')),
            # Cas du point suivi directement par un autre signe
            (r'\.(?=[,;!?])', '¤'),
        ]

        for pattern, repl in substitutions:
            text = re.sub(pattern, repl, text, flags=re.MULTILINE)

        return text

    text = mask(text)
    phrases = re.findall('.*?[.\n]', text)

    def punct_back(text):
        return text.replace('¤', '.')

    phrases = [punct_back(p).strip() for p in phrases if p.strip()]

    return phrases

"""    def compare(self, model, sentence, n) :
        if not sentence.vectorized : 
            sentence.vectorized = model.encode([sentence.content])
        
        source_vector = sentence.vectorized
        text_matrix = self.vectorize()
        similarities = cosine_similarity([source_vector], text_matrix)
        best_similarities_index = np.argsort(similarities).flip() #Les indices des documents les plus similaires, dans l'ordre
        
        return best_similarities_index[0:n]


        class Vectorized_states(dict) :
    #Class conçue uniquement pour pouvoir rendre hashable le states_set
    def __init__(self) :
        super()
    
    def __getitem__(self, key):
        return self.__dict__[tuple(sorted(key))]
    
    def __setitem__(self, key, value):
        self.__dict__[tuple(sorted(key))] =  value

    def __contains__(self, key) :
        return tuple(sorted(key)) in self.__dict__
"""