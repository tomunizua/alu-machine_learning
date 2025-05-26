#!/usr/bin/env python3
"""
This module deals with the uni-gram BLEU score.
"""


import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    """
    references_length = []
    length = len(sentence)
    words = {}

    for translation in references:
        references_length.append(len(translation))
        for word in translation:
            if word in sentence and word not in words.keys():
                words[word] = 1

    total = sum(words.values())
    index = np.argmin([abs(len(i) - length) for i in references])
    best_match = len(references[index])

    if length > best_match:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(best_match) / float(length))
    BLEU_score = BLEU * np.exp(np.log(total / length))

    return BLEU_score
