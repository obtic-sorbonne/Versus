from Global_stuff import *
from Document import Document
from os import walk, path
from copy import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np


class Corpus : 
    """A l'échelle du corpus"""
    def __init__(self, path : str = None) :
        if path :
            documents_paths = []
            for (dirpath, _, filenames) in walk(path):
                documents_paths.extend([path.join(dirpath, doc_name) for doc_name in filenames if doc_name.endswith(".txt")])
                if Global_stuff.DEPTH == 0 : 
                    break;
            self.documents = [Document(path) for path in documents_paths]

        else : 
            self.documents = []
        
        self.tf_idf_updated = TfidfVectorizer()

    ###!!UTILITY ET CORPUS MANAGEMENT
    def add_doc(self, document : Document) : 
        if self.index(document) == -1 :
            self.documents.append(document)
            return f"Document {document.name} ajouté"
        else : 
            return f"Impossible d'ajouter \"{document.name}\", un document du même nom est déjà présent dans le corpus"

    def get_doc_by_name(self, name) : 
        return self.documents[self.get_documents_names().index(name)]
    
    def get_documents_names(self) : 
        return [d.name for d in self.documents]
    
    def index(self, document) : 
        """Renvoie l'indice du document s'il est présent, -1 sinon"""
        names = self.get_documents_names()
        if document.name in names :  
            doc_index = names.index(document.name)
        else : 
            doc_index = -1
        return doc_index
    
    def __len__(self) : 
        return len(self.documents)

    def __add__(self, corpus_2) : 
        corpus_sum = Corpus()

        corpus_sum.documents = self.documents + [d for d in corpus_2.documents if d.name not in self.get_documents_names()]
        
        return corpus_sum

    def __sub__(self, document, inplace = False) :
        """Renvoie le corpus sans le document, s'il y est"""
        
        if inplace : 
            corpus_sub = self
        else : 
            corpus_sub = self.copy()

        if isinstance(document, Document) :
            index = self.index(document)
        elif isinstance(document, int) : 
            index = document

        if index != -1 :
            corpus_sub.documents.pop(index)
        
        return corpus_sub

    def copy(self) : 
        corpus_copy = Corpus()
        corpus_copy.documents = copy(self.documents)

        return corpus_copy



    def filter(self, query) :
        corpus = Corpus()
        #Le nombre d'occurence de la query dans les documents
        score = [len(d.text.find_words(query)) for d in self.documents]
        #classement = np.flip(np.argsort(score))

        #On le récupère dans l'ordre de la query la plus à la moins présente
        #new_docs = []
        #for c in classement : 
            #On s'arrête au premier document qui ne contient plus la query
        #    if score[c] == 0 : 
        #        break;
        #    new_docs.append(self.documents[c])

        corpus.documents = [self.documents[i] for i in range(len(self)) if score[i]>0]

        return corpus
  
    ###!!COMPARAISON ET VECTORISATION : 

    def vectorize_corpus(self, model) :
        
        corpus_matrix = np.array([doc.vectorize_document(model, self.tf_idf_updated) for doc in self.documents])
        
        return corpus_matrix

    def compare(self, source : Document, model=SentenceTransformer(Global_stuff.MODEL_NAME), n=0, inplace=True) : 
        self.tf_idf_updated.fit([d.text.origin_content for d in self.documents])

        if n<=0 or n> len(self) :
            n = len(self)

        #On copie le corpus au cas où le document ne se trouveraient pas dedans
        corpus = self.copy()

        if isinstance(source, int) : 
            index = source

        elif isinstance(source, Document) : 
            index = corpus.index(source)
            if corpus.index(source) == -1 : 
                corpus.add_doc(source)

        #On récupère la vectorisation (une matrice)
        corpus_matrix = corpus.vectorize_corpus(model)

        #On compare le cosinus du vecteur source au reste de la matrice
        source_vector = corpus_matrix[index].copy()
        if source_vector.shape[0] > 1 : #Si le vecteur n'a qu'une seule dimension, on l'empaquette
            source_vector = [source_vector]

        similarities = cosine_similarity(source_vector, corpus_matrix)[0]
        best_similarities_index = np.argsort(similarities) #Les indices des documents du plus petit cosinus au plus grand
        best_similarities_index = np.flip(best_similarities_index) #On l'inverse pour commencer par les plus similaires

        best_docs = [corpus.documents[i] for i in best_similarities_index]

        sorted_corpus = Corpus()
        sorted_corpus.documents = best_docs[0:n]

        if inplace : 
            self.documents = sorted_corpus.documents


        self.project(model)

        return sorted_corpus

    def project(self, model) :
        
        corpus_matrix = self.vectorize_corpus(model)
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(corpus_matrix)

        plt.figure(figsize=(10, 8))
        for i, name in enumerate([d.name for d in self.documents]):
            x, y = reduced[i]
            plt.scatter(x, y)
            plt.text(x + 0.01, y + 0.01, name, fontsize=12)

        plt.title("Projection PCA des embeddings de document")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()