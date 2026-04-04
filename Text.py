"""
Classe Text pour la manipulation et segmentation de texte.
"""

from Global_stuff import Global_stuff
from Chunks import Sentence, Word
from math import ceil
import numpy as np
from re import search
from itertools import accumulate
import re


class Text:
    """Manipulation textuelle et segmentation."""

    def __init__(self, name: str, content: str):
        self.name = name
        self.origin_content = content.strip().replace("\r", "\n")
        
        matches = list(re.finditer(r'\b\w+\b', self.origin_content))
        self.words = [Word(match.start(), match.end(), match.group()) for match in matches]
        self.vectorized = None
        self.default()

    def __getitem__(self, c):
        if isinstance(c, tuple):
            a, b = c[0], c[1]
            return self.origin_content.replace("\n", " ")[a:b]
        raise TypeError("Please get with a couple")

    def __repr__(self):
        return self.origin_content

    def default(self):
        """Réinitialise le contenu."""
        self.content = self.origin_content
        matches = list(re.finditer(r'\b\w+\b', self.origin_content))
        self.words = [Word(match.start(), match.end(), match.group()) for match in matches]
        self.update_sentences()
        return self

    def n_grams(self, n=1):
        return [self.words[i:i+n] for i in range(len(self.words)-n+1)]

    def remove_stopwords(self):
        stopwords = Global_stuff.STOPWORDS
        word_contents = [word.content.lower() for word in self.words]
        temp = [i for i, w in enumerate(word_contents) if w not in stopwords]
        self.words = [self.words[i] for i in temp]
        self.content = " ".join([w.content for w in self.words]).strip()
        return self

    def vectorize(self, model):
        if self.vectorized is None:
            sentences_content = [s.content for s in self.sentences]
            self.vectorized = model.encode(sentences_content)
        return self.vectorized
    
    def vectorize_window(self, model, w=1):
        if w == 1:
            return self.vectorize(model), [[s] for s in self.sentences]
        elif w > 1:
            step = ceil(w/2)
            sentences_windows = [self.sentences[i:i+w] for i in range(0, self.n_sentences, step)]
            sentences_windows_content = [" ".join([str(s) for s in window]) for window in sentences_windows]
            windows_matrix = model.encode(sentences_windows_content)
            return windows_matrix, sentences_windows
    
    def find_words(self, regex):
        matches = re.finditer(regex, self.origin_content, re.IGNORECASE)
        return [(m.start(), m.end()) for m in matches]        

    def update_sentences(self):
        sentences = split_sentences(self.content)
        to_delete = set()

        for i, e in enumerate(sentences):
            # Supprimer les fragments sans lettre (guillemets seuls, ponctuation…)
            if search('[a-zA-ZÀ-öø-ÿ]', e) is None:
                to_delete.add(i)
                continue
            # Supprimer les fragments trop courts sans mot d'au moins 2 caractères
            words_in = re.findall(r'\b\w{2,}\b', e)
            if not words_in:
                to_delete.add(i)
                continue
            if i not in to_delete and len(e) < Global_stuff.MIN_SENT_LENGTH and i != len(sentences) - 1:
                to_delete.add(i)
                sentences[i + 1] = e + sentences[i + 1]

        for i in sorted(to_delete, reverse=True):
            del sentences[i]

        # Calcul des positions RÉELLES dans origin_content via str.find()
        # La borne de fin est start + len(content) — pas le début de la phrase suivante
        origin = self.origin_content
        result = []
        cursor = 0
        for s in sentences:
            key = s[:40]
            pos = origin.find(key, cursor)
            if pos == -1:
                # fallback : cherche sans le début pour les cas de normalisation
                pos = origin.find(s[:20].strip(), cursor)
            if pos == -1:
                pos = cursor
            end = pos + len(s)
            result.append((pos, end, s))
            cursor = end  # avance après la fin de la phrase, jamais en arrière

        self.sentences = [Sentence(start, content) for start, end, content in result]
        for i, (start, end, _) in enumerate(result):
            self.sentences[i]._end = end

        self.n_sentences = len(self.sentences)
        return self


def split_sentences(text):
    """Segmentation en phrases avec masquage des faux points."""
    def mask(text):
        substitutions = [
            (r'—([^—]*?)[.;!?](.*?)\n', r'—\1¤\2\n'),
            (r'"([^"]*?)[.;!?](.*?)"', r'"\1¤\2"'),
            (r'"([^"]*?)[.;!?](.*?)"', r'"\1¤\2"'),
            (r'«([^»]*?)[.;!?](.*?)»', r'«\1¤\2»'),
            (r"'([^']*?)[.;!?](.*?)'", r"'\1¤\2'"),
            (r'\(([^\)]*?)[.;!?](.*?)\)', r'\(\1¤\2\)'),
            (r'\{([^\}]*?)[.;!?](.*?)\}', r'\{\1¤\2\}'),
            (r'\[([^\]]*?)[.;!?](.*?)\]', r'\[\1¤\2\]'),
            (r'\b\d+\.\d+\b', lambda m: m.group(0).replace('.', '¤')),
            (r'[\w\.-]+@[\w\.-]+\.\w+', lambda m: m.group(0).replace('.', '¤')),
            (r'(?i)(https?:\/\/|w{3}\.)[\w.%\/?=&#:;+-]{5,}', lambda m: m.group(0).replace('.', '¤')),
            (r'(?i)\b(etc|cf|ex|dr|mr|mrs|mlle|mme|ms|st|mt|vs|vol|chap|fig|al|ibid|op\.cit|loc\.cit|n\.b|e\.g|i\.e|ca)\.', lambda m: m.group(0).replace('.', '¤')),
            (r'\b(M)\.', lambda m: m.group(0).replace('.', '¤')),
            (r'(?i)\b(prof|univ|coll|col|vol|réf|ref|tabl|graph|edit|nat|reg|suppl|tom|act|gen|lib|acad|comm)\.(\d+)?', lambda m: m.group(0).replace('.', '¤')),
            (r'\b([\u0621-\u064A])\.', r'\1¤'),
            (r'(?i)\b([IVXLCDM]+)\.', r'\1¤'),
            (r'\b[a-zA-Z]\.', r'\g<0>'.replace('.', '¤')),
            (r'^\d+\.', lambda m: m.group(0).replace('.', '¤')),
            (r'\.(?=[,;!?])', '¤'),
        ]

        for pattern, repl in substitutions:
            text = re.sub(pattern, repl, text, flags=re.MULTILINE)
        return text

    text = mask(text)
    phrases = re.findall('.*?[.\n]', text)
    phrases = [p.replace('¤', '.').strip() for p in phrases if p.strip()]
    return phrases
