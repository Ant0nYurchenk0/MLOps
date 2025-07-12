import os
import re
import boto3
from dotenv import load_dotenv
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
import numpy as np

load_dotenv()

sb = SnowballStemmer("english")
lc = LancasterStemmer()
ps = PorterStemmer()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_USERNAME = os.getenv("MINIO_USERNAME", "minio")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD", "minio123")
EMBEDDINGS_BUCKET = os.getenv("EMBEDDINGS_BUCKET", "embeddings")
WIKI_EMBED_KEY = os.getenv("WIKI_EMBED_KEY", "wiki-news-300d-1M.vec.json")
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_USERNAME,
    aws_secret_access_key=MINIO_PASSWORD,
)

# print("Downloading words from MinIO…")
WORDS = s3.get_object(Bucket=EMBEDDINGS_BUCKET, Key=WIKI_EMBED_KEY)


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    "Generate possible spelling corrections for word."
    return known([word]) or known(edits1(word)) or [word]


def known(words):
    "The subset of words that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from word."
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


# Use fast text as vocabulary
def words(text):
    return re.findall(r"\w+", text.lower())


def P(word):
    "Probability of word."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return -WORDS.get(word, 0)


def edits2(word):
    "All edits that are two edits away from word."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def singlify(word):
    return "".join(
        [letter for i, letter in enumerate(word) if i == 0 or letter != word[i - 1]]
    )


def stream_embeddings(obj_body):
    for raw_line in obj_body.iter_lines():
        try:
            # Try UTF-8 first, then fall back to latin-1 for binary files
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("latin-1")
            
            if len(line) > 100:  # Skip header or malformed lines
                parts = line.strip().split()
                if len(parts) < 2:  # Need at least word and one coefficient
                    continue
                word = parts[0]
                parts_cleaned = []
                for token in parts[1:]:  # Skip the word, only process coefficients
                    try:
                        parts_cleaned.append(float(token))
                    except ValueError:
                        parts_cleaned.append(0.0)
                
                if len(parts_cleaned) > 0:  # Only yield if we have coefficients
                    coefs = np.asarray(parts_cleaned, dtype="float32")
                    yield word, coefs
        except Exception as e:
            # Skip malformed lines
            continue
