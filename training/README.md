# Stage 1 - Training

Task - Manually classify product reviews into 2 classes (positive / negative).

## How?

- Goto <http://amazon.in>. Copy the [ASIN][asin] of ~30 random products into a
  file (one ASIN per line).

- Install dependencies.

        pip3 install -r requirements.txt

- Execute `review_classifier` given the ASIN file as an argument classifying
  each review as **p**ositive or **n**egative.

      ./review_classifier apn.txt

- Final data is in [`data.txt`](data.txt).

[asin]: https://en.wikipedia.org/w/index.php?title=Amazon_Standard_Identification_Number&oldid=705247351
