#!/usr/bin/env python3
"""
Converts a gensim word2vec model to a keras Embedding layer
"""

def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
