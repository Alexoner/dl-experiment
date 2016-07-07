import numpy as np
from gensim.models import word2vec
from tqdm import trange

from ..utils import *

def trainW2V(corpus, epochs=1, embed_size=100,
             min_word_count=1, num_workers=8,
             window=10, sample=1e-5):
    """
    @args
    corpus: list of list of tokens
    """
    global word2vec
    word2vec = word2vec.Word2Vec(workers=num_workers,
                                 sample=sample,
                                 size=embed_size,
                                 min_count=min_word_count,
                                 window=window)
    np.random.shuffle(corpus)
    word2vec.build_vocab(corpus)
    for epoch in trange(epochs):
        np.random.shuffle(corpus)
        word2vec.train(corpus)
        word2vec.alpha *= 0.9
        word2vec.min_alpha = word2vec.alpha
    print("Training word2vec done.")
    return word2vec
