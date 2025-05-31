# File: preprocess_service.py

import os
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from spacy.cli import download as spacy_download
from tqdm import tqdm
from threading import Lock


# ---------- Request / Response Schemas ----------
class BuildVocabRequest(BaseModel):
    texts: List[str]  # list of all texts (train + test)
    max_length: int


class BuildVocabResponse(BaseModel):
    word_dict: Dict[str, int]
    lemma_dict: Dict[str, str]
    sequences: List[List[int]]  # padded/truncated sequences of indices


class SequenceRequest(BaseModel):
    text: str
    max_length: int
    word_dict: Dict[str, int]


class SequenceResponse(BaseModel):
    sequence: List[int]


# ---------- FastAPI App ----------
app = FastAPI(
    title="Preprocessing Microservice", docs_url="/docs", openapi_url="/openapi.json"
)

# Load SpaCy model (download if missing)
try:
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "tagger"])
except OSError:
    spacy_download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "tagger"])

# In‐memory caching (thread‐safe) to avoid rebuilding for identical batches
cache_lock = Lock()
vocab_cache: Dict[str, BuildVocabResponse] = {}
# Keyed by a hash of sorted(texts) + max_length; value is BuildVocabResponse


# ---------- Helper Functions ----------
def build_vocab_and_sequences(texts: List[str], max_length: int) -> BuildVocabResponse:
    """
    Build word_dict + lemma_dict over `texts`, then tokenize and convert each text to a padded sequence.
    """
    # 1) Build dictionaries
    word_dict: Dict[str, int] = {}
    lemma_dict: Dict[str, str] = {}
    word_index = 1
    sequences: List[List[int]] = []

    # 2) Process in streaming fashion
    docs = nlp.pipe(texts, batch_size=32)
    for doc in tqdm(docs, total=len(texts), desc="Building vocab & sequences"):
        seq: List[int] = []
        for token in doc:
            if token.pos_ == "PUNCT":
                continue
            text = token.text
            if text not in word_dict:
                word_dict[text] = word_index
                lemma_dict[text] = token.lemma_
                word_index += 1
            seq.append(word_dict[text])
        sequences.append(seq)

    # 3) Pad / truncate all sequences
    #    We will pad with zeros at the end ("post"), and cut off any tokens beyond max_length.
    padded_seqs: List[List[int]] = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded = seq[:max_length]
        else:
            padded = seq + [0] * (max_length - len(seq))
        padded_seqs.append(padded)

    return BuildVocabResponse(
        word_dict=word_dict, lemma_dict=lemma_dict, sequences=padded_seqs
    )


# ---------- Endpoints ----------
@app.post("/build_vocab", response_model=BuildVocabResponse)
def build_vocab_endpoint(request: BuildVocabRequest):
    # Simple cache key: JSON‐serialize sorted texts + max_length
    cache_key = json.dumps(
        {"texts": request.texts, "max_length": request.max_length}, sort_keys=True
    )
    with cache_lock:
        if cache_key in vocab_cache:
            return vocab_cache[cache_key]

    resp = build_vocab_and_sequences(request.texts, request.max_length)
    with cache_lock:
        vocab_cache[cache_key] = resp
    return resp


@app.post("/sequence", response_model=SequenceResponse)
def sequence_endpoint(request: SequenceRequest):
    # Convert a single text to sequence using provided word_dict
    seq: List[int] = []
    doc = nlp(request.text)
    for token in doc:
        if token.pos_ == "PUNCT":
            continue
        idx = request.word_dict.get(token.text, 0)
        seq.append(idx)

    # Pad/truncate to max_length
    if len(seq) >= request.max_length:
        padded = seq[: request.max_length]
    else:
        padded = seq + [0] * (request.max_length - len(seq))

    return SequenceResponse(sequence=padded)
