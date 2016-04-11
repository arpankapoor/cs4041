from collections import Counter


def get_bigram_frequencies(strings):
    """
    Return the count of all bigrams that occur in the list of strings.

    Args:
        strings: list of strings.

    Returns:
        A Counter with
            - key: "word1 word2"
            - value: frequency of the bigram in the given list of strings
    """
    counter = Counter()

    for string in strings:
        words = string.split()
        bigrams = [' '.join(bigram) for bigram in zip(words[:-1], words[1:])]
        counter.update(bigrams)

    return counter


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


def get_combined_frequencies(strings, min_bigram_occur=3):
    """
    Return the count of all words/bigrams such a word is counted only if
    the count of the bigrams in which that word occurs < min_bigram_occur.

    eg. for a string 'A B C', the word B is counted if and only if
        1. count('A B') < min_bigram_occur
        2. count('B C') < min_bigram_occur

    Args:
        strings: A list of strings.
        min_bigram_occur: minimum number of occurrences of a bigram so as to
                          skip its constituent words' count.

    Returns:
        A Counter with
            - key: word/bigram
            - value: frequency of the word/bigram in the given list of strings
    """
    def make_bigram(word1, word2):
        return ' '.join([word1, word2])

    counter = Counter()
    bigram_counter = get_bigram_frequencies(strings)

    # Add the bigram counts
    counter.update(bigram_counter)

    for string in strings:
        words = string.split()
        no_of_words = len(words)
        for i, word in enumerate(words):
            curr_bigrams = []
            if i >= 1:
                curr_bigrams.append(make_bigram(words[i-1], words[i]))
            if i < no_of_words - 1:
                curr_bigrams.append(make_bigram(words[i], words[i+1]))

            if all([bigram_counter[bigram] < min_bigram_occur
                    for bigram in curr_bigrams]):
                counter.update([word])

    return counter


def get_vocabulary_from_frequencies(counters, min_occur=1):
    """
    Return a set of words/bigrams which occur at least min_occur times,
    given a list of word/bigram frequencies.

    Args:
        counters: A list of Counters with
                  - key: word/bigram
                  - value: frequency of the word/bigram in the
                           given list of strings
        min_occur: minimum number of occurrences of a word/bigram for it to be
                   included in the vocabulary.

    Returns:
        A set of words/bigrams.
    """
    master_counter = Counter()
    vocabulary = set()

    for counter in counters:
        master_counter = master_counter + counter

    for string, count in master_counter.items():
        if count >= min_occur:
            vocabulary.add(string)

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
