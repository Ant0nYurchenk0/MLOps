from tqdm import tqdm
import numpy as np
from helpers import ps, lc, sb, correction


def load_para(word_dict, lemma_dict, embeddings_index):

    embed_size = 300
    nb_words = len(word_dict) + 1
    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.0
    for key in tqdm(word_dict):
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                # Pad with zeros if too short
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lemma_dict[key]
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Handle dimension mismatch by truncating or padding
            if len(embedding_vector) > embed_size:
                embedding_vector = embedding_vector[:embed_size]
            elif len(embedding_vector) < embed_size:
                padded_vector = np.zeros(embed_size, dtype=np.float32)
                padded_vector[:len(embedding_vector)] = embedding_vector
                embedding_vector = padded_vector
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        if len(key) > 1:
            word = correction(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Handle dimension mismatch by truncating or padding
                if len(embedding_vector) > embed_size:
                    embedding_vector = embedding_vector[:embed_size]
                elif len(embedding_vector) < embed_size:
                    padded_vector = np.zeros(embed_size, dtype=np.float32)
                    padded_vector[:len(embedding_vector)] = embedding_vector
                    embedding_vector = padded_vector
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
        embedding_matrix[word_dict[key]] = unknown_vector
    return embedding_matrix, nb_words
