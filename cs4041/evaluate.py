from collections import defaultdict
import math
import random
from .classify import calculate_accuracy
from .train import train, train_with_bigram_features


def join_reviews(parts):
    """
    Join the given list of reviews into a single part.

    Args:
        parts: a list of dictionaries with
                        - key: class
                        - value: list of reviews belonging to this class.

    Returns:
        A single part with the same type as parts[0].
    """
    reviews = defaultdict(list)

    for part in parts:
        for class_, review_list in part.items():
            reviews[class_].extend(review_list)

    return reviews


def split_reviews(reviews, no_of_parts=10):
    """
    Split the given reviews into x equal parts (each containing
    equal number of reviews from each class).

    Args:
        reviews: dictionary with
                 - key: class
                 - value: list of reviews belonging to this class.
        no_of_parts: number of parts to divide into.

    Returns:
        List of parts with each element having the same type
        as the argument `reviews`.
    """
    # list of parts
    parts = []

    # key: class
    # value: number of reviews of this class in each part
    reviews_per_part = {}

    shuffled_reviews = defaultdict(list)

    for class_, review_list in reviews.items():
        reviews_per_part[class_] = len(review_list) / no_of_parts
        shuffled_reviews[class_] = review_list
        random.shuffle(shuffled_reviews[class_])

    # Split into parts
    for i in range(no_of_parts):
        part = defaultdict(list)

        for class_, review_list in shuffled_reviews.items():
            start_idx = math.floor(i * reviews_per_part[class_])
            end_idx = math.floor((i+1) * reviews_per_part[class_])

            part[class_] = review_list[start_idx: end_idx]

        parts.append(part)

    return parts


def evaluate(reviews, stopwords, min_occur=2, no_of_parts=10):
    """
    Do a no_of_parts-fold cross evaluation.

    1. Split the given reviews into x equal parts (each containing
       equal number of reviews from each class).

    2. For each part:
        1. Train with the remaining parts.
        2. Classify the review texts in this part using
           the above trained model.
        3. Calculate the accuracy of this trained model.

    3. Return a list of the accuracies of all trained models.

    Args:
        reviews: dictionary with
                 - key: class
                 - value: list of reviews belonging to this class.
        stopwords: list of words to remove from all documents.
        min_occur: minimum number of occurrences of a word for it to be
                   included in the vocabulary.
        no_of_parts: number of parts to divide into.

    Returns:
        List of accuracies of all models trained.
    """
    parts = split_reviews(reviews, no_of_parts)

    # Accuracy of each model
    accuracies = [0] * no_of_parts

    # Train with all parts except i and classify the reviews in part i
    for i in range(no_of_parts):
        joined_part = join_reviews([
            parts[j] for j in range(no_of_parts) if j != i
        ])

        trained_model = train(joined_part, stopwords, min_occur)

        accuracies[i] = calculate_accuracy(parts[i], stopwords, trained_model)

    return accuracies


def evaluate_with_bigram_features(reviews, stopwords,
                                  min_occur=2, no_of_parts=10):
    """
    Similar to the above function except that this function also takes into
    account some bigram features.

    Args:
        reviews: dictionary with
                 - key: class
                 - value: list of reviews belonging to this class.
        stopwords: list of words to remove from all documents.
        min_occur: minimum number of occurrences of a word/bigram for it to be
                   included in the vocabulary.
        no_of_parts: number of parts to divide into.

    Returns:
        List of accuracies of all models trained.
    """
    parts = split_reviews(reviews, no_of_parts)

    # Accuracy of each model
    accuracies = [0] * no_of_parts

    # Train with all parts except i and classify the reviews in part i
    for i in range(no_of_parts):
        joined_part = join_reviews([
            parts[j] for j in range(no_of_parts) if j != i
        ])

        trained_model = train_with_bigram_features(joined_part, stopwords,
                                                   min_occur)

        accuracies[i] = calculate_accuracy(parts[i], stopwords, trained_model)

    return accuracies
