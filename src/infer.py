import random

import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig

import hydra
from cold_cnn import ColdCNN, OffensiveLangDetector
from cold_data import ColdDataset, ColdVectorizer, load_data
from utils import Cols, create_tensors_from_dataset, generate_skip_gram_comments

SAMPLE_SIZE = 3


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    processed_data = cfg.datasets.processed
    X_col = cfg.features.X_col
    y_col = cfg.features.y_col

    # Load test data
    test_data = load_data(processed_data.test, X_col, y_col)

    # Take a sample of test data
    sample_ids = random.sample(range(len(test_data)), SAMPLE_SIZE)
    test_samples = [test_data[id] for id in sample_ids]

    # Load fitted COLD model
    model = OffensiveLangDetector(
        model=ColdCNN(cfg.model, cfg.features.word_vec_dim, cfg.features.max_seq_len),
        learning_rate=cfg.model.learning_rate,
    )
    model.load_state_dict(torch.load(cfg.model_file))
    model.eval()

    vectorizer = ColdVectorizer()

    torch.manual_seed(seed=42)

    for _, sample in enumerate(test_samples):

        comment, label = sample
        logger.info(f"Original comment: {comment}")
        logger.info(f"Groundtruth label: {label}")

        # Synthesize permuted comments with one token skipped
        permuted_comments, skipped_tokens = generate_skip_gram_comments(
            vectorizer, comment
        )
        permuted_data = [(comment, label) for comment in permuted_comments]
        cold_permuted = ColdDataset(permuted_data, vectorizer)

        output = create_tensors_from_dataset(cold_permuted, cfg.features.max_seq_len)
        logger.debug(output["vectors"].size())

        with torch.no_grad():
            logits = model(output["vectors"])

        logits = torch.squeeze(logits).numpy().T
        logger.info(f"Predicted logit for the original comment: {logits[0]}")

        res = pd.DataFrame(
            data={
                Cols.comment: permuted_comments,
                Cols.skipped_token: skipped_tokens,
                Cols.label: output["label"].numpy(),
                Cols.logit: logits,
            }
        )
        res[Cols.importance] = res[Cols.logit].apply(lambda logit: logits[0] - logit)

        # Sort permuted comments by the importance of the skipped tokens
        sorted_res = res[1:].sort_values(by=Cols.importance, ascending=False)

        logger.info(f"\n{str(sorted_res)}\n")


if __name__ == "__main__":
    main()
