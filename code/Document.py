#from Global_stuff import *
import numpy as np
from Text import Text
import chardet

class Document : 
    """A l'échelle d'un document : sa vectorisation, comparaison, et son contenu"""
    def __init__(self, name, file_value=None, content=None) :
        self.name = name
        self.vectorized = None
        self.vector_sentences = None

        if file_value : 
            encoding = chardet.detect(file_value)['encoding']
            self.text = Text(self.name, file_value.decode(encoding))
        elif content : 
            self.text = Text(self.name, content)

    def vectorize_document(self, model, vectorizerTfIdf) :
        if self.vectorized is None : 
            
            #Init
            t = self.text
            sentences_content = [s.content for s in t.sentences]
            
            #On vectorise les phrases
            if self.vector_sentences is None :
                self.vector_sentences = model.encode(sentences_content)
                print(f"Document {self.name} vectorisé.")
            
            #On associe aux phrases leur vecteur
            for i in range(t.n_sentences) : 
                if t.sentences[i].vectorized is None :
                    t.sentences[i].vectorized = self.vector_sentences[i]
            
            #On crée un tf.idf pour la vectorisation hybride
            
            
            #Idée : On renforce les phrases qui contiennent des mots que les autres phrases ne contiennent pas, car ce sont elles qui caractérisent le document

            #On regarde le poids qu'occupe la phrase dans ce document
            tf_idf = vectorizerTfIdf.transform(sentences_content)
            weights = np.sum(tf_idf, axis=1) #Le poids total des occurences des mots de chaque phrase, c'est à dire à quel point elles sont présente dans le texte
            weights = list(map(lambda x : float(x), weights))
            lengths_weights = np.array([weights[i]/len(sentences_content[i].split(" ")) for i in range(t.n_sentences)]) #On divise le poids par le nombre de mots pour normaliser
            
            self.vectorized = np.mean(self.vector_sentences * lengths_weights[:, np.newaxis], axis=0) #On pondère chaque vecteur par son score

            print(f"Document {self.name} traité en hybride")
        
        return self.vectorized
"""    
    def compare(self, corpus, mode, n, parameters) : 
        
        #On crée un corpus identique auquel on ajoute le document source
        if n<=0 or n>= len(corpus) :
            n = len(corpus)

        corpus = corpus.copy()
        index = corpus.contains(self)
        if corpus.contains(self) == -1 : 
            corpus.add_doc(self)

        #On récupère la vectorisation (une matrice)
        corpus_matrix = corpus.vectorize(mode, parameters)

        #On compare le cosinus du vecteur source au reste de la matrice
        source_vector = corpus_matrix[index] 

        corpus_matrix = np.delete(corpus_matrix, index, axis=0) #On ne compare pas le document à lui même

        similarities = cosine_similarity(source_vector, corpus_matrix)
        best_similarities_index = np.argsort(similarities).flip() #Les indices des documents les plus similaires, dans l'ordre
        
        return best_similarities_index[0:n]
"""  