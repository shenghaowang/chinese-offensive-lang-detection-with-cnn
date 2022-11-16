from typing import List, Tuple

import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig

import hydra
from cold_cnn import ColdCNN, OffensiveLangDetector
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

    # Initialise ABSA model
    trainer(
        model=ColdCNN(hyparams=cfg.model, seq_len=cfg.features.max_seq_len),
        feature_params=cfg.features,
        hyparams=cfg.model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        # model_file=cfg.model_file,
    )


def trainer(
    model: ColdCNN,
    feature_params: DictConfig,
    hyparams: DictConfig,
    train_data: List[Tuple],
    valid_data: List[Tuple],
    test_data: List[Tuple],
    # model_file: str,
) -> None:
    # Create a pytorch trainer
    trainer = pl.Trainer(max_epochs=hyparams.max_epochs, check_val_every_n_epoch=1)

    # Initialize our data loader with the passed vectorizer
    data_module = ColdDataModule(
        vectorizer=ColdVectorizer(),
        batch_size=feature_params.batch_size,
        max_seq_len=feature_params.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    # Instantiate a new model
    old = OffensiveLangDetector(
        model,
        learning_rate=hyparams.learning_rate,
    )

    # Train and validate the model
    trainer.fit(
        old,
        data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    # Test the model
    trainer.test(old, data_module.test_dataloader())

    # Predict on the same test set to show some output
    output = trainer.predict(old, data_module.test_dataloader())

    for i in range(2):
        logger.info("====================")
        logger.info(f"Comment: {output[1]['comments'][i]}")
        logger.info(f"Prediction: {output[1]['predictions'][i].numpy()}")
        logger.info(f"Actual Label: {output[1]['label'][i].numpy()}")

    # Export fitted model
    # model_dir = Path(model_file).parent
    # model_dir.mkdir(parents=True, exist_ok=True)

    # torch.save(model.state_dict(), model_file)
    # logger.info(f"ABSA model exported to {model_file}.")


if __name__ == "__main__":
    main()