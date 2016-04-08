from collections import Counter


def get_word_frequencies(strings):
    """
    Return the count of all words that occur in the list of strings.

    Args:
        strings: list of strings.

    Returns:
        A Counter with
            - key: word
            - value: frequency of the word in the given list of strings
    """
    joined_string = ' '.join(string for string in strings)

    return Counter(joined_string.split())


def get_vocabulary_from_word_frequencies(counters, min_occur=1):
    """
    Return a set of words which occur at least min_occur times,
    given a list of word frequencies.

    Args:
        counters: A list of Counters with
                  - key: word
                  - value: frequency of the word in the given list of strings
        min_occur: minimum number of occurrences of a word for it to be
                   included in the vocabulary.

    Returns:
        A set of words.
    """
    master_counter = Counter()
    vocabulary = set()

    for counter in counters:
        master_counter = master_counter + counter

    for word, count in master_counter.items():
        if count >= min_occur:
            vocabulary.add(word)

    return vocabulary


def get_vocabulary(strings, min_occur=1):
    """
    Return a set of words which occur at least min_occur times in
    the list of strings.

    Args:
        strings: list of strings.
        min_occur: minimum number of occurrences of a word for it to be
                   included in the vocabulary.

    Returns:
        A set of words.
    """
    counter = get_word_frequencies(strings)

    return get_vocabulary_from_word_frequencies([counter], min_occur)


def read_vocabulary(fname):
    """
    Read the list of words in the vocabulary from a
    file with given filename, one word on a line.

    Args:
        fname: filename.

    Returns:
        A list of words.
    """
    with open(fname) as f:
        vocabulary = {line.strip() for line in f}

    return vocabulary


def write_vocabulary(vocabulary, fname):
    """
    Write the list of words in the vocabulary to a
    file with given filename, one word on a line.

    Args:
        vocabulary: list of words.
        fname: filename.
    """
    with open(fname, 'w') as f:
        for word in vocabulary:
            f.write(word + '\n')
