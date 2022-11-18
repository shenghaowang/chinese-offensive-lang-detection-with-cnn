from typing import List

import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import f1_score

import hydra
from cold_cnn import ColdCNN, OffensiveLangDetector
from cold_data import ColdDataModule, ColdVectorizer, load_data


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    processed_data = cfg.datasets.processed
    X_col = cfg.features.X_col
    y_col = cfg.features.y_col

    # Load training data
    train_data = load_data(processed_data.train, X_col, y_col)
    valid_data = load_data(processed_data.dev, X_col, y_col)
    test_data = load_data(processed_data.test, X_col, y_col)

    # Load fitted ABSA model
    model = OffensiveLangDetector(
        model=ColdCNN(hyparams=cfg.model, seq_len=cfg.features.max_seq_len),
        learning_rate=cfg.model.learning_rate,
    )
    model.load_state_dict(torch.load(cfg.model_file))
    model.eval()

    # Make predictions for training data
    data_module = ColdDataModule(
        vectorizer=ColdVectorizer(),
        batch_size=cfg.features.batch_size,
        max_seq_len=cfg.features.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )
    trainer = pl.Trainer(max_epochs=cfg.model.max_epochs, check_val_every_n_epoch=1)
    output = trainer.predict(model, data_module.train_dataloader())

    # Collect predicted logits
    logits = []
    labels = []
    for batch_output in output:
        batch_logits = [logit.item() for logit in batch_output["logits"]]
        batch_labels = [label.item() for label in batch_output["label"]]
        logits.extend(batch_logits)
        labels.extend(batch_labels)

    logger.info(f"Number of logits: {len(logits)}")
    logger.info(f"Number of labels: {len(labels)}")

    logger.info(f"First 10 labels: {labels[:10]}")
    logger.info(f"First 10 logits: {logits[:10]}")

    # Search for the optimal threshold
    search_for_threshold(labels, logits)


def search_for_threshold(labels: List[int], logits: List[str]):
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


if __name__ == "__main__":
    main()
