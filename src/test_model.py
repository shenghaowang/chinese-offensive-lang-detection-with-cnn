import torch
from loguru import logger
from omegaconf import DictConfig
from torchsummary import summary

import hydra
from cold_cnn import ColdCNN
from cold_data import ColdDataModule, ColdVectorizer, load_data


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg: DictConfig):
    processed_data = cfg.datasets.processed
    X_col = cfg.features.X_col
    y_col = cfg.features.y_col

    # Load training, validation, and test data
    train_data = load_data(processed_data.train, X_col, y_col)
    valid_data = load_data(processed_data.dev, X_col, y_col)
    test_data = load_data(processed_data.test, X_col, y_col)

    logger.info(f"Volume of training data: {len(train_data)}")
    logger.info(f"Volume of validation data: {len(valid_data)}")
    logger.info(f"Volume of test data: {len(test_data)}")

    # Create data loader
    device = torch.device("cpu")
    data_module = ColdDataModule(
        vectorizer=ColdVectorizer(),
        batch_size=cfg.features.batch_size,
        max_seq_len=cfg.features.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )
    training_generator = data_module.train_dataloader()

    local_batch = next(iter(training_generator))
    local_vectors = local_batch["vectors"].to(device)
    local_labels = local_batch["label"].to(device)
    logger.info(f"Batch data for training: {local_vectors.size()}")
    logger.info(f"Batch labels for training: {local_labels.size()}")

    model = ColdCNN(cfg.model, cfg.features.max_seq_len)
    # cnn_out = model.conv(local_vectors)
    # logger.info(f"cnn_output: {cnn_out.data.size()}")

    # relu_out = model.relu(cnn_out)
    # pooling_out = model.pooling(relu_out)
    # logger.info(f"pooling_out: {pooling_out.data.size()}")

    # flatten_out = model.flatten(pooling_out)
    # logger.info(f"flatten_out: {flatten_out.data.size()}")

    # fc_out = model.fc(flatten_out)
    # logger.info(f"fc_out: {fc_out.data.size()}")

    summary(model, (1, cfg.features.max_seq_len, cfg.features.word_vec_dim))


if __name__ == "__main__":
    main()
