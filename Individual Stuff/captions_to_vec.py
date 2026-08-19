import string
import numpy as np
import math
from collections import Counter
from coco_data import COCOData
from gensim.models import KeyedVectors
from pathlib import Path

filename = Path(__file__).parent / "data" / "glove.6B.200d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(filename, binary = False)

def tokenize(text): 
        text_clean = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text_clean.lower().split()
        return tokens
    
class CaptionVectorizer:
    def __init__(self, coco: COCOData):
        all_captions = coco.get_all_captions()
        self.all_caps_tokens = [tokenize(captions) for captions in all_captions]
        self.N_captions = len(self.all_caps_tokens)

        #precompute, once, how many captions contain each word
        self.doc_counts = Counter()
        for caption_tokens in self.all_caps_tokens:
            for word in set(caption_tokens):
                self.doc_counts[word] += 1

        #cache vectors so a repeated caption isn't re-embedded every batch/epoch
        self._cache = {}


    def compute_idf(self, caption): # caption should be a list of token (tokens), and the function computes the idf for each corresponding token and returns as a list
        tokens = (tokenize(caption) if isinstance(caption, str) else caption)

        idf_list = []
        for token in tokens:
            n_t = self.doc_counts.get(token, 0)
            #storing the calculuted (base 10 log) idf val in score,
            score = math.log10(self.N_captions/n_t) if n_t > 0 else 0.0
            idf_list.append(score)

        return idf_list

    def __call__(self, caption):
        if caption in self._cache:
            return self._cache[caption]

        tokens = tokenize(caption)
        #vector of the caption
        w_cap = np.zeros(200, dtype = np.float32)
        
        idfs = self.compute_idf(tokens)
        
        if(len(idfs)!=len(tokens)):
            print("Error, the size of idf does not match the size of tokens")
        
        for i in range(len(tokens)):
            if tokens[i] not in glove:
                continue
            glove_vec = glove[tokens[i]]
            idf = idfs[i]
            w_cap += idf * glove_vec

        norm = np.linalg.norm(w_cap)

        if norm > 1e-8: #almost 0
            w_hat_cap = w_cap/norm

        else:
            w_hat_cap = np.zeros(200, dtype = np.float32)

        self._cache[caption] = w_hat_cap
        return w_hat_cap
