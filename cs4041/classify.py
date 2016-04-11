from .stopwords import rm_stopwords


def calculate_cond_probability(review_text, class_, trained_model):
    """
    Calculate the log probability of class given review times the review
    probability.

    P(c | d) * P(d) = P(d | c)* P(c)
                    = P(w_1 | c) * ... * P(w_n | c) * P(c)

    Args:
        review: a string.
        class_: given class (+/-).
        trained_model: A 2-tuple
            1. class probabilities: a dictionary with
                                    - key: class
                                    - value: log probability of that class
            2. likelihoods: a dictionary with
                            - key: tuple (word/bigram, class)
                            - value: log(likelihood of word/bigram given class)

    Return:
        Log((probability of class given review) * (probability of review))
    """
    class_probability = trained_model[0]
    likelihood = trained_model[1]

    ans = class_probability[class_]
    words = review_text.split()
    no_of_words = len(words)

    idx = 0
    while idx < no_of_words:
        curr_bigram = None
        if idx < no_of_words-1:
            curr_bigram = ' '.join([words[idx], words[idx+1]])

        # Add bigram probability OR word probability OR unknown probability
        # in that order
        if curr_bigram is not None and (curr_bigram, class_) in likelihood:
            ans = ans + likelihood[(curr_bigram, class_)]
            idx = idx + 2
        else:
            if (words[idx], class_) in likelihood:
                ans = ans + likelihood[(words[idx], class_)]
            else:
                ans = ans + likelihood[('<UNKNOWN>', class_)]
            idx = idx + 1

    return ans


def classify_review(review_text, stopwords, trained_model):
    """
    Given a review, classify it into +/- using given trained model.

    Args:
        review: a string.
        stopwords: list of words to remove from all reviews.
        trained_model: A 2-tuple
            1. class probabilities: a dictionary with
                                    - key: class
                                    - value: log probability of that class
            2. likelihoods: a dictionary with
                            - key: tuple (word/bigram, class)
                            - value: log(likelihood of word/bigram given class)

    Return:
        '+' or '-'
    """
    ans = '+'
    prob = {}

    # remove stop words
    cleaned_review = rm_stopwords([review_text], stopwords)[0]

    for class_ in ['+', '-']:
        prob[class_] = calculate_cond_probability(cleaned_review, class_,
                                                  trained_model)

    if prob['-'] > prob['+']:
        ans = '-'

    return ans


def calculate_accuracy(reviews, stopwords, trained_model):
    """
    Calculate the accuracy of classification of all reviews (after stop word
    removal), using the given trained model.

                     tp + tn
    Accuracy = -------------------
                fp + tp + tn + fn

    where,
        tp = number of true positives
        tn = number of true negatives
        fp = number of false positives
        fn = number of false negatives

    Args:
        reviews: dictionary with
                 - key: class
                 - value: list of reviews belonging to this class.
        stopwords: list of words to remove from all reviews.
        trained_model: A 2-tuple
            1. class probabilities: a dictionary with
                                    - key: class
                                    - value: log probability of that class
            2. likelihoods: a dictionary with
                            - key: tuple (word/bigram, class)
                            - value: log(likelihood of word/bigram given class)

    Return:
        Accuracy of the model in classifying given reviews.
    """
    tp, tn, fp, fn = [0] * 4

    for class_, review_list in reviews.items():
        for review_text in review_list:
            predicted_class = classify_review(review_text, stopwords,
                                              trained_model)

            if class_ == '+' and predicted_class == '+':
                tp = tp + 1
            elif class_ == '-' and predicted_class == '-':
                tn = tn + 1
            elif class_ == '-' and predicted_class == '+':
                fp = fp + 1
            elif class_ == '+' and predicted_class == '-':
                fn = fn + 1

    return (tp + tn) / (tp + tn + fp + fn)
