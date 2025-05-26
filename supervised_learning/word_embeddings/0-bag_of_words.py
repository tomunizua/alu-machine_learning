#!/usr/bin/env python3
"""
this module contains the function bag_of_words
it creates a word embedding matrix using the bag of words
method
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    '''
    creates a bag of words embedding matrix
    '''
    vectorizer = CountVectorizer(vocabulary=vocab)
    X_train_counts = vectorizer.fit_transform(sentences)
    embeddings = X_train_counts.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
