import re
from .vocabulary import read_vocabulary


def read_stopwords(fname):
    """
    Read the list of stop words in the file with given filename,
    one word on a line.

    Args:
        fname: filename.

    Returns:
        A list of stop words.
    """
    return read_vocabulary(fname)


def rm_stopwords(strings, stopwords):
    """
    Return a list of strings after processing as follows:
        1. Remove all non alphabetic characters (except whitespace).
        2. Convert to lower case.
        3. Remove stop words (also replacing multiple whitespace characters
           with a single whitespace).

    Args:
        strings: list of strings to be processed.
        stopwords: list of words to remove from given strings.

    Returns:
        A list of strings.
    """
    newlist = []

    for string in strings:
        # Remove non-alphabetic characters except whitespace
        newstring = re.sub(r'[^a-zA-Z\s]+', '', string)

        newstring = newstring.lower()

        # Remove stop words
        newstring = ' '.join([word for word in newstring.split()
                              if word not in stopwords])

        if len(newstring) > 0:
            newlist.append(newstring)

    return newlist
