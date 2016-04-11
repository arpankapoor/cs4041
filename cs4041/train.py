from collections import Counter, defaultdict
import math
import pickle
from .stopwords import rm_stopwords
from .vocabulary import (get_combined_frequencies,
                         get_word_frequencies,
                         get_vocabulary_from_frequencies)


def get_class_probabilities(documents):
    """
    Calculate log probabilities of all classes from given
    list of classified documents.

               | documents(c) |
    P(c) = ------------------------
           | total # of documents |

    Args:
        documents: dictionary with
                   - key: class
                   - value: list of documents belonging to this class.

    Returns:
        A dictionary with
            - key: class
            - value: log probability of this class
    """
    # key: class
    # value: number of documents of this class
    class_doc_count = {}

    class_probability = {}
    total_doc_count = 0

    for class_, doclist in documents.items():
        curr_class_doc_count = len(doclist)
        class_doc_count[class_] = curr_class_doc_count
        total_doc_count = total_doc_count + curr_class_doc_count

    for class_ in documents.keys():
        class_probability[class_] = (math.log(class_doc_count[class_]) -
                                     math.log(total_doc_count))

    return class_probability


def get_word_likelihoods(documents, min_occur=2):
    """
    Calculate the likelihood of a word given a class for all words that
    occur at least min_occur times and all classes (with add-one smoothing).

                           count(w_k, c_j) + 1
    P(w_k | c_j) = ------------------------------------
                     ___
                     \                      |   |
                     /    (count(w, c_j)) + | V | + 1
                     ---                    |   |
                      w

    Note: +1 in the denominator for the '<UNKNOWN>' word.

    Args:
        documents: dictionary with
                   - key: class
                   - value: list of documents belonging to this class.
        min_occur: minimum number of occurrences of a word for it to be
                   included in the vocabulary.

    Returns:
        A dictionary with
            - key: (word, class)
            - value: log likelihood of word given class
    """
    # key: class
    # value: Counter for all words in that class
    word_frequency = defaultdict(Counter)

    # key: (word, class)
    # value: log likelihood of word given class
    word_likelihood = {}

    for class_, doclist in documents.items():
        word_frequency[class_] = get_word_frequencies(doclist)

    # Create vocabulary from the word frequencies of all classes
    vocabulary = get_vocabulary_from_frequencies(
        [word_frequency[class_] for class_ in documents.keys()],
        min_occur
    )

    vocabulary_size = len(vocabulary)

    for class_ in documents.keys():
        # Sum of word frequencies in a class (summation count(w, c))
        class_count = sum(word_frequency[class_].values())

        word_likelihood[('<UNKNOWN>', class_)] = (math.log(1) -
                                                  math.log(class_count +
                                                           vocabulary_size +
                                                           1))

        for word in vocabulary:
            # count(word, class) + 1
            num = word_frequency[class_][word] + 1

            # (summation count(word, class)) + |V| + 1
            den = class_count + vocabulary_size + 1

            word_likelihood[(word, class_)] = math.log(num) - math.log(den)

    return word_likelihood


def get_likelihoods_with_bigram_features(documents, min_occur=3):
    """
    Calculate the likelihood of a word/bigram given a class for all
    words/bigrams that occur at least min_occur times and all
    classes (with add-one smoothing).

                           count(t_k, c_j) + 1
    P(t_k | c_j) = ------------------------------------
                     ___
                     \                      |   |
                     /    (count(t, c_j)) + | V | + 1
                     ---                    |   |
                      t

    Note: +1 in the denominator for the '<UNKNOWN>' word.

    Args:
        documents: dictionary with
                   - key: class
                   - value: list of documents belonging to this class.
        min_occur: minimum number of occurrences of a word/bigram for it to be
                   included in the vocabulary.

    Returns:
        A dictionary with
            - key: (word/bigram, class)
            - value: log likelihood of word/bigram given class
    """
    # key: class
    # value: Counter for all words/bigrams in that class
    frequency = defaultdict(Counter)

    # key: (word, class)
    # value: log likelihood of word/bigram given class
    likelihood = {}

    for class_, doclist in documents.items():
        frequency[class_] = get_combined_frequencies(doclist)

    # Create vocabulary from the word frequencies of all classes
    vocabulary = get_vocabulary_from_frequencies(
        [frequency[class_] for class_ in documents.keys()],
        min_occur
    )

    vocabulary_size = len(vocabulary)

    for class_ in documents.keys():
        # Sum of word frequencies in a class (summation count(w, c))
        class_count = sum(frequency[class_].values())

        likelihood[('<UNKNOWN>', class_)] = (math.log(1) -
                                             math.log(class_count +
                                                      vocabulary_size + 1))

        for token in vocabulary:
            # count(word, class) + 1
            num = frequency[class_][token] + 1

            # (summation count(word, class)) + |V| + 1
            den = class_count + vocabulary_size + 1

            likelihood[(token, class_)] = math.log(num) - math.log(den)

    return likelihood


def train(documents, stopwords, min_occur=2):
    """
    Train a multinomial naive Bayes classifier based on given
    list of classified documents.

    Args:
        documents: dictionary with
                   - key: class
                   - value: list of documents belonging to this class.
        stopwords: list of words to remove from all documents.
        min_occur: minimum number of occurrences of a word for it to be
                   included in the vocabulary.


    Returns:
        A 2-tuple:
        1. class probabilities: a dictionary with
                                - key: class
                                - value: log probability of that class
        2. word likelihoods: a dictionary with
                             - key: tuple (word, class)
                             - value: log(likelihood of word given class)
    """
    newdocs = defaultdict(list)
    for class_, doclist in documents.items():
        newdocs[class_] = rm_stopwords(doclist, stopwords)

    return (get_class_probabilities(newdocs),
            get_word_likelihoods(newdocs, min_occur))


def train_with_bigram_features(documents, stopwords, min_occur=3):
    """
    Train a multinomial naive Bayes classifier based on given
    list of classified documents, with bigram features considered.

    Args:
        documents: dictionary with
                   - key: class
                   - value: list of documents belonging to this class.
        stopwords: list of words to remove from all documents.
        min_occur: minimum number of occurrences of a word/bigram for it to be
                   included in the vocabulary.


    Returns:
        A 2-tuple:
        1. class probabilities: a dictionary with
                                - key: class
                                - value: log probability of that class
        2. likelihoods: a dictionary with
                        - key: tuple (word/bigram, class)
                        - value: log(likelihood of word/bigram given class)
    """
    newdocs = defaultdict(list)
    for class_, doclist in documents.items():
        newdocs[class_] = rm_stopwords(doclist, stopwords)

    return (get_class_probabilities(newdocs),
            get_likelihoods_with_bigram_features(newdocs, min_occur))


def read_trained_model(fname):
    """
    Read the trained model from a python pickle file.

    Args:
        fname: file name.

    Returns:
        The trained model saved.
    """
    with open(fname, 'rb') as f:
        model = pickle.load(f)

    return model


def write_trained_model(model, fname):
    """
    Save the given model to a python pickle file.

    Args:
        model: model to save.
        fname: file name.
    """
    with open(fname, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
