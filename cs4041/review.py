from bs4 import BeautifulSoup
from collections import defaultdict
import random
import requests
import urllib.parse


def get_amazon_reviews(asin, tld, class_=None, limit=10):
    """
    Fetch customer reviews for any Amazon product.

    Args:
        asin: Amazon Standard Identification Number.
        tld: top level domain of the Amazon store for this ASIN.
        limit: maximum number of reviews to fetch.
        class_: class of reviews to fetch => +/-/None

    Returns:
        A list of reviews.

    Raises:
        HTTPError
        ValueError
    """
    review_base_url = "http://www.amazon.{TLD}/product-reviews/"
    valid_tlds = ['com.au', 'com.br', 'ca', 'cn', 'fr', 'de', 'in',
                  'it', 'co.jp', 'com.mx', 'nl', 'es', 'co.uk', 'com']

    if tld not in valid_tlds:
        raise ValueError('{} is not a valid Amazon TLD.'.format(tld))

    url = urllib.parse.urljoin(review_base_url.format(TLD=tld), asin)
    params = {'pageNumber': 1}
    reviews = []

    if class_ == '+':
        params['filterByStar'] = "positive"
    elif class_ == '-':
        params['filterByStar'] = "critical"

    while limit > 0:
        resp = requests.get(url, params=params)

        # Try again if get a 503
        if resp.status_code == 503:
            continue

        # Raise an error if not a 503 and no reviews have been collected.
        if len(reviews) == 0:
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        review_soup = soup.find_all(class_='review-text', limit=limit)

        # Break if this page has no reviews
        if len(review_soup) == 0:
            break

        curr_reviews = [' '.join(review.stripped_strings)
                        for review in review_soup]

        reviews.extend(curr_reviews)

        limit = limit - len(curr_reviews)
        params['pageNumber'] = params['pageNumber'] + 1

    return reviews


def read_reviews(fname):
    """
    Read reviews from a file with given filename in the following format:

        CLASS<TAB>REVIEW

    Args:
        fname: name of the file to read from.

    Returns:
        A dictionary with
            - key: class
            - value: list of reviews belonging to this class.
    """
    reviews = defaultdict(list)

    with open(fname) as f:
        for line in f:
            class_, review_text = line.strip().split(sep='\t', maxsplit=1)
            reviews[class_].append(review_text)

    return reviews


def write_reviews(reviews, fname):
    """
    Write the reviews to a file with given filename in the following format:

        CLASS<TAB>REVIEW

    Args:
        reviews: dictionary with
                 - key: class
                 - value: list of reviews belonging to this class.
        fname: name of the file to write to.
    """
    with open(fname, 'w') as f:
        for class_, review_texts in reviews.items():
            # Randomize the review texts
            random.shuffle(review_texts)

            for review_text in review_texts:
                if len(review_text) > 0:
                    f.write(class_ + '\t' + review_text + '\n')
