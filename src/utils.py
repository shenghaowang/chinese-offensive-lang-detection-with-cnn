from typing import List

from loguru import logger
from sklearn.metrics import f1_score

from cold_data import ColdVectorizer


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
) -> List[str]:
    """Generate all possible variant comments with a token skipped

    Parameters
    ----------
    comment : str
        the original processed comment

    Returns
    -------
    List[str]
        list of variant comments
    """
    all_tokens = cold_vectorizer.tokenize(comment)
    num_tokens = len(all_tokens)

    comments = []
    comments.append(comment)

    for i in range(num_tokens):
        skip_gram_tokens = all_tokens[:i] + all_tokens[(i + 1) :]  # noqa: E203
        comments.append("".join(skip_gram_tokens))

    return comments
