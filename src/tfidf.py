from nltk.tokenize import sent_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from scipy.spatial.distance import cosine
import numpy as np
import re
from nltk import word_tokenize, pos_tag
from os import walk


def get_filenames(dir, depth=None):
    files     = []
    if depth == None:
        for path, dirs, fnames in walk(dir):
            files.extend([path + '/' + f for f in fnames])
    else:
        i = 0
        for path, dirs, fnames in walk(dir):
            files.extend([path + '/' + f for f in fnames])
            if i >= depth:
                break
            i += 1
    return files

def read_file(file):
    text = None
    with tf.io.gfile.GFile(file) as f:
        text = f.read()
    f.close()
    return text

def get_corpus(dir, depth=None):
    files  = get_filenames(dir, depth)
    corpus = [read_file(file) for file in files]
    return corpus

def cosine_similarity(x, y):
    return -(cosine(x, y) - 1.)

class TFIDFSummarizer(object):

    def __init__(self, corpus):
        """
        Arguments
        ---------
            corpus : list of strings.
        """
        self.tokenizer = Tokenizer()
        lexicon        = []
        # Remove stop words from our corpus for tokenizer.
        for i in range(len(corpus)):
            words     = text_to_word_sequence(corpus[i])
            words     = [word for word in words if word not in STOP_WORDS]
            lexicon.append(' '.join(words))

        self.tokenizer.fit_on_texts(lexicon)

    def __call__(self, text, k=5):
        """
        Arguments
        ---------
            text : string (text) to be summarized.
            k    : soft limit for sentence length of summary.

        Returns
        -------
            summary : summary of the input text
        """
        sentences = sent_tokenize(text)
        length    = len(sentences)
        cleaned   = []
        for i in range(length):
            words = text_to_word_sequence(sentences[i])
            words = [word for word in words if word not in STOP_WORDS]
            cleaned.append(' '.join(words))

        tfidfs = self.tokenizer.texts_to_matrix(cleaned, mode='tfidf')
        scores = []
        for i in range(length):
            # Cosine similarity of sentence with every other sentence.
            score = [cosine_similarity(tfidfs[i], tfidfs[j]) for j in range(length) if i != j]
            scores.append(score)

        scores  = np.array(scores).mean(axis=1)
        indices = np.argpartition(scores, kth=k)
        indices = indices[-(k+1):]
        i_preps = set(indices.tolist())
        # To give context, we add the previous sentence of each sentence to the summary.
        for i in indices:
            if i-1 >= 0:
                i_preps.add(i-1)

        indices = sorted(list(i_preps))

        summary = [sentences[j] for j in indices]
        summary = '\n\t' + ' '.join(summary)
        return summary


if __name__ == "__main__":
    i          = 2
    path       = "texts"

    corpus     = get_corpus(path)
    text       = corpus[i]

    summarizer = TFIDFSummarizer(corpus)
    summary    = summarizer(text)

    print(summary)