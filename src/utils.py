from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence

from cold_data import ColdDataset, ColdVectorizer


@dataclass
class Cols:
    comment: str = "comment"
    skipped_token: str = "skipped_token"
    label: str = "label"
    logit: str = "logit"
    importance: str = "importance"


def search_for_threshold(labels: List[int], logits: List[str]):
    """Search for optimal threshold for generating binary
    predictions. F1 score will be used to guide the search.

    Parameters
    ----------
    labels : List[int]
        groundtruth labels
    logits : List[str]
        logits predicted by model
    """
    best_f1 = 0
    optimal_threshold = 0
    for i in range(5, 96):
        threshold = i / 100
        preds = [1 if logit >= threshold else 0 for logit in logits]
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold

    logger.info(f"Optimal threshold = {optimal_threshold} is found.")
    logger.info(f"Best F1 = {best_f1}")


def generate_skip_gram_comments(
    cold_vectorizer: ColdVectorizer, comment: str
) -> Tuple[List[str], List[str]]:
    """_summary_

    Parameters
    ----------
    cold_vectorizer : ColdVectorizer
        vectorizer
    comment : str
        the original processed comment

    Returns
    -------
    Tuple[List[str], List[str]]
        list of variant comments, list of skipped tokens
    """
    all_tokens = cold_vectorizer.tokenize(comment)
    all_tokens = [token.text for token in all_tokens]
    num_tokens = len(all_tokens)

    comments = []
    skipped_tokens = []
    comments.append(comment)
    skipped_tokens.append("")  # indicates no skipped token

    for i in range(num_tokens):
        skip_gram_tokens = all_tokens[:i] + all_tokens[(i + 1) :]  # noqa: E203
        comments.append("".join(skip_gram_tokens))
        skipped_tokens.append(all_tokens[i])

    return comments, skipped_tokens


def create_tensors_from_dataset(ds: ColdDataset, max_seq_len: int):
    # Convert word vectors to tensors
    word_vector = [torch.Tensor(item["vectors"]) for item in ds]

    # Trim sequences to ensure consistent length
    word_vector = [
        torch.nn.ZeroPad2d((0, 0, 0, max_seq_len - len(vec)))(vec)
        if max_seq_len > len(vec)
        else vec[:max_seq_len, :]
        for vec in word_vector
    ]

    labels = torch.LongTensor(np.array([item["label"] for item in ds]))

    # Pad each vector sequence to the same size
    # [batch_size, word_vec_dim, sequence_length]
    padded_word_vector = pad_sequence(word_vector, batch_first=True).transpose(1, 2)

    return {
        "vectors": padded_word_vector,
        "label": labels,
        "comments": [item["comment"] for item in ds],
    }
