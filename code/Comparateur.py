import numpy as np
from Global_stuff import *

from textdistance import hamming

import rapidfuzz
from rapidfuzz import fuzz

from copy import deepcopy

from difflib import SequenceMatcher

from scipy.sparse import coo_array, vstack

class PairText :
    def __init__(self, text1, text2) :
        self.text1 = text1
        self.text2 = text2
    """
    def compare_sentences(self, n=10, w=1) :
        Compare les deux textes par ensemble de phrases. Un w élevé (4 ou 5, par exemple) prendra plus de temps comparera des paragraphes, et un w petit (1 ou 2) des phrases entre elles.
        M_t1, sentences_1 = self.text1.vectorize_window(w=w)
        M_t2, sentences_2 = self.text2.vectorize_window(w=w)
        #On calcule la similarité
        similarities = cosine_similarity(M_t1, M_t2)

        #On argsort en étalant la matrice sur un seul axe
        coords = np.flip(np.argsort(similarities, axis=None))       
        best_sentences = []

        for c in coords[:n] :
            #On recrée les coordonnées matricielles
            i,j = np.unravel_index(c, similarities.shape)
            s1 = sentences_1[i]
            s2 = sentences_2[j]
            #On ajoute les deux phrases à la liste
            best_sentences.append((s1, s2))

        return best_sentences"""
    
    def compare_and_export_to_csv(self, n=3, score_threshold=0.9, diff=False, separator=";", path="comparaison.csv") : 
        result = self.compare_n_grams(n=n, score_threshold=score_threshold, diff=diff)
        csv_text = self.text1.name + ";" + self.text2.name +"\n" #Une colonne par livre
        for pair in result : 
            csv_text += "\""+ self.text1[pair[0]] + "\"" #premier passage
            csv_text += separator #séparation
            csv_text += "\"" + self.text2[pair[1]] + "\"" #deuxième passage
            csv_text += "\n" #bout de ligne

        with open(path, "w") as f : 
            f.write(csv_text)

        return csv_text

    def compare_n_grams(self, n=3, model=SentenceTransformer(Global_stuff.MODEL_NAME), score_threshold=0.9, diff=True) :
        """Comparaison  par défaut des paires de n_grams, """ 
        
        result = self.compare_ngrams_model(n=n, model=model, score_threshold=score_threshold)
        
        if diff :
            for i, pair in enumerate(result) : 
                p_t1 = pair[0] #Intervalle similaire du texte 1
                s_1 = p_t1[0] #Coordonnées de départ de l'intervalle 1
                p_t2 = pair[1] #Intervalle similaire du texte 2
                s_2 = p_t2[0] #Coordonnées de départ de l'intervalle 2
                
                t1 = self.text1[p_t1]
                t2 = self.text2[p_t2]

                result[i] = (pair[0], pair[1], [], [])

                s = SequenceMatcher(lambda x : x == " ", t1, t2)
                
                for action, i1, i2, j1, j2 in s.get_opcodes() : #Coordonnées des différences dans le SequenceMatcher
                    if action != 'equal' : 
                        result[i][2].append((s_1 + i1, s_1 + i2)) #On les met à leur place dans le vrai texte
                        result[i][3].append((s_2 + j1, s_2 + j2))
        else : 
            for i, pair in enumerate(result) :
                result[i] = (pair[0], pair[1], [], [])
        
        print(result)
        return result

    def remove_stopwords_texts(self) : 
        for t in (self.text1, self.text2) : 
            t.remove_stopwords()

    def set_default_texts(self) : 
        for t in (self.text1, self.text2) : 
            t.default()
    
    def show_sim(self, indexes) : 
        res = ""
        for i, ind in enumerate(indexes) : 
            print(i, " : ", self.text1[ind[0]], "|||", self.text2[ind[1]] )
            res += f"{i} : {self.text1[ind[0]]} ||| {self.text2[ind[1]]}\n"
        return res

    def compare_ngrams_model(self, n=3, model=SentenceTransformer(Global_stuff.MODEL_NAME), score_threshold=0.93) : 
        """Compare les N_grams d'un texte à l'autre à l'aide d'un modèle de LLM appliqué sur les N_grams et d'une similarité cosinus.
        Ensuite, aggrègue
        Enfin, renvoie les coordonnées dans les texte des paires de N_grams similaires.
        """
            #Calcul des scores----------- 
        #On construit des n-grams de mots, par liste de l'objet "Mot"
        n_grams1, n_grams2 = self.text1.n_grams(n), self.text2.n_grams(n)

        #On crée le contenu de chaine de caractères qui vont être comparées
        content_1 = [" ".join([w.content for w in n_words]) for n_words in n_grams1]
        content_2 = [" ".join([w.content for w in n_words]) for n_words in n_grams2]

        #Encodage des textes : scaling linéaire par texte
        M_1 = model.encode(content_1)
        print(self.text1.name, " modelisé")
        M_2 = model.encode(content_2)
        print(self.text2.name, " modelisé")

        #Calcul du cosinus : scaling au carré
        scores = None
        step = 3000
        for i in range(0, M_1.shape[0], step) : 
            print(i,"/",M_1.shape[0])
            mini_m = cosine_similarity(M_1[i:i+step], M_2)
            mini_m[mini_m<score_threshold] = 0
            
            if scores is None :
                scores = coo_array(mini_m)    
            else :
                scores = vstack([scores, coo_array(mini_m)])

        print(len(scores.data))

        #Aggrégation----------- 
        #Pour aggréger en respectant tous les cas (similarité multiples, aggrégations en chaînes, etc...) il faut se servir de la matrice et regarde la case en haut à gauche.
        #La case diagonale haut gauche : les n_grams de ces coordonnées précède d'un seul mot les n_grams actuels, pour chaque texte
        #Si elle est dans la matrice (donc très similaire également), on aggrègue le tout.
        data = scores.data.tolist()
        aggreg = dict()
        to_remove = []

        row_and_col = list(zip(scores.row, scores.col))
        #Boucle globale d'aggrégation
        for i, (row_ind, col_ind) in enumerate(row_and_col) : #On regarde les coordonnées non nulles
            row_ind, col_ind = int(row_ind), int(col_ind)

            g1, g2 = n_grams1[row_ind], n_grams2[col_ind]

            aggreg[(row_ind, col_ind)] = (g1, g2, data[i]) #On ajoute dans le dictionnaire les n_grams correspondantes
            
            if i == 0 : #Cas du premier élément : impossible à aggréger car il n'y a rien avant 
                pass
            #Cas de la case haut gauche
            #Si cette case est non nulle, il faut aggréger.
            elif (row_ind-1, col_ind-1) in row_and_col : 

                #Soit l'aggrégation correspondante à la case en question (qui est non nulle donc ajoutée)
                prec = aggreg[(row_ind-1, col_ind-1)]

                #On actualise la case actuelle en lui donnant : 
                new_gram1 = prec[0][0:-n+1] + g1 #Comme n_gram1 : les mots de prec[0] (le ngram précédent du texte1) qui ne sont pas déjà dedans
                new_gram2 = prec[1][0:-n+1] + g2 #Comme n_gram1 : les mots de prec[1] (le ngram précédent du texte2)  qui ne sont pas déjà dedans
                new_score = (prec[2]+data[i])/2 #Comme score, la moyenne avec le score précédent 
                aggreg[(row_ind, col_ind)] = (new_gram1, new_gram2, new_score)
                
                print("MODIFYING FROM", row_ind-1, col_ind-1)

                #On pensera ensuite à retirer le n_gram précédent, puisqu'il s'est fait aggréger.
                to_remove.append((row_ind-1, col_ind-1))

            """
            else : 
                if (row_ind-1, col_ind) in row_and_col : #Si la case du haut ressemble
                    prec = aggreg[(row_ind-1, col_ind)]

                    new_gram1 = prec[0][0:-n+1] + g1
                    new_score = (prec[2]+data[i])/2
                    
                    aggreg[(row_ind, col_ind)] = (new_gram1, g2, new_score) #Même chose, mais en aggrégeant seulement les lignes
                    to_remove.append((row_ind-1, col_ind))

                if (row_ind, col_ind-1) in row_and_col : #Si la case de gauche ressemble 
                    prec = aggreg[(row_ind, col_ind-1)]

                    new_gram2 = prec[1][0:-n+1] + g2
                    new_score = (prec[2]+data[i])/2

                    aggreg[(row_ind, col_ind)] = (g1, new_gram2, new_score) #Même chose, mais en aggrégeant seulement les colonnes
                    to_remove.append((row_ind, col_ind-1))"""

        
        #On supprime du dictionnaire tout ce qui est en double car présent en lui même et dans la version aggregée qui lui succède
        for i in to_remove : 
            aggreg.pop(i)

        #Aggreg est un dictionnaire de la forme : 
            #coord du n-gram en X (ligne), coord du n-gram en Y (colonne) : ngram1 aggregé, ngram2 aggregé, score 

            #Résultat----------
        #On récupère les n_grams et les scores
        best_n_grams = zip([item[0] for _, item in aggreg.items()], [item[1] for _, item in aggreg.items()], [item[2] for _, item in aggreg.items()])

        #On les trie par score puis par indice de mot croissant depuis le texte source
        best_n_grams = sorted(best_n_grams, key = lambda t : (t[1][0].start))

        #On récupère les coordonnées des meilleurs n_grams
        best_n_grams_indexes = []
        for b in best_n_grams : 
            n_words1 = b[0]
            n_words2 = b[1]
            #L'objet Mot contient son indice de départ dans le texte d'origine
            best_n_grams_indexes.append(((n_words1[0].start, n_words1[-1].end), (n_words2[0].start, n_words2[-1].end)))   
        self.show_sim(best_n_grams_indexes)
        print(best_n_grams_indexes)


        #Objectif : renvoyer une meilleure structure de données, plus adaptées à montrer les ressemblances entre passage qui prend en compte aggrégation et réutilisation
        return best_n_grams_indexes

    def find_words(self, words) :
        return self.text1.find_words(words), self.text2.find_words(words)

    def find_sentences(self, sentence, n) :
        return self.text1.find_sentences(sentence, n), self.text2.find_sentences(sentence, n)
    
    """
    def compare_ngrams_fuzzy(self, n=3, score_threshold = 0.93) :

        score_threshold = score_threshold*100

        #On construit des n-grams de mots, par liste de l'objet "Mot"
        n_grams1, n_grams2 = self.text1.n_grams(n), self.text2.n_grams(n)
        
        #On crée le contenu de chaine de caractères qui vont être comparées
        content_1 = [" ".join([w.content for w in n_words]) for n_words in n_grams1]
        content_2 = [" ".join([w.content for w in n_words]) for n_words in n_grams2]

        #On compare toutes les paires, scaling au carré
        scores = rapidfuzz.process.cdist(content_1, content_2, scorer=fuzz.token_set_ratio, processor = lambda x : x.lower(), workers=-1)
        scores = np.array(scores)
        scores[scores<=93] = 0 #On ne garde que celles supérieures à 93 (arbitraire)
        
        #On transforme les données en une matrice creuse
        scores = coo_array(scores)

        #On transforme en nparray pour l'accessibilité
        n_grams1 = np.array(n_grams1)
        n_grams2 = np.array(n_grams2)
        tuples = zip(n_grams1[scores.row], n_grams2[scores.col], scores.data)

        #On trie les coordonnées par indice croissant et score décroissant 
        best_n_grams = sorted(tuples, key=lambda x : (x[2]), reverse=True)

        best_n_grams_indexes = []

        #On récupère les coordonnées des meilleurs n_grams
        for b in best_n_grams : 
            n_words1 = b[0]
            n_words2 = b[1]
            #L'objet Mot contient son indice de départ dans le texte d'origine
            best_n_grams_indexes.append(((n_words1[0].start, n_words1[-1].end), (n_words2[0].start, n_words2[-1].end)))

        return best_n_grams_indexes"""