from bs4 import BeautifulSoup
import requests
import urllib.parse

BASE_REVIEW_URL = "http://www.amazon.in/product-reviews/"
DEFAULT_COUNT = 10


def get_reviews(asin, count=DEFAULT_COUNT, review_class=None):
    """Fetch customer reviews for any amazon.in product.

    Args:
        asin: Amazon Standard Identification Number.
        count: Number of reviews to fetch.
        review_class: Positive or Negative (p or n)

    Returns:
        A list of reviews. Number of reviews <= count.

    Raises:
        HTTPError
    """
    url = urllib.parse.urljoin(BASE_REVIEW_URL, asin)
    params = {'pageNumber': 1}
    reviews = []

    review_class = review_class.lower()

    if review_class == 'p':
        params['filterByStar'] = "positive"
    elif review_class == 'n':
        params['filterByStar'] = "critical"

    while count > 0:
        resp = requests.get(url, params=params)

        if len(reviews) == 0:
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        review_soup = soup.find_all(class_='review-text', limit=count)

        if len(review_soup) == 0:
            break

        reviews.extend([' '.join(review.stripped_strings)
                        for review in review_soup])

        count = count - len(review_soup)
        params['pageNumber'] = params['pageNumber'] + 1

    return reviews
