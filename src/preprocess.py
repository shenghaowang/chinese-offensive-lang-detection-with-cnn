import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from stopwordsiso import stopwords

import hydra

ZH_STOPS = stopwords("zh")


@dataclass
class Cols:
    raw_text: str = "TEXT"
    processed_text: str = "processed_text"
    seq_len: str = "len"
    label: str = "label"


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    raw_data = cfg.datasets.raw
    processed_data = cfg.datasets.processed
    rm_stops = cfg.datasets.rm_stopwords

    df = pd.read_csv(raw_data.train)
    logger.info(f"Training data: {df.shape}")
    logger.debug(f"\n{df.head()}")

    # Check the class distribution
    logger.info(
        f"Class distribution of the training data:\n{df['label'].value_counts()}"
    )

    # Check the length of the comments
    df[Cols.seq_len] = df[Cols.raw_text].apply(len)
    logger.info(f"Length of comments:\n{df['len'].describe()}")

    for percentile in [50, 75, 95]:
        logger.info(
            f"{percentile}th percentile of the comment length: "
            + f"{np.percentile(df[Cols.seq_len].values, percentile)}"
        )

    # Preview text processing
    logger.debug("Preview the processed text:")
    for text in df[Cols.raw_text][:5]:
        logger.debug(f"Original text: {text}")
        logger.debug(f"Processed text: {pre_process(text, rm_stops)}")
        logger.debug("\n")

    # Create directory for storing the processed data
    output_dir = Path(processed_data.train).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # To reduce the chance of data drift, combine the given datasets
    # and resplit for training, validation, and test
    combined_df = pd.concat(
        [pd.read_csv(raw_data[ds_type]) for ds_type in ["train", "dev", "test"]]
    )
    combined_df[Cols.processed_text] = combined_df[Cols.raw_text].apply(pre_process)
    train_df, rest_df = train_test_split(combined_df, test_size=0.3, random_state=42)
    dev_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)

    logger.info(f"Volume of training data: {len(train_df)}")
    logger.info(f"Volume of dev data: {len(dev_df)}")
    logger.info(f"Volume of test data: {len(test_df)}")

    # Export data
    required_cols = [Cols.processed_text, Cols.label]
    write_processed_data(train_df[required_cols], processed_data["train"])
    write_processed_data(dev_df[required_cols], processed_data["dev"])
    write_processed_data(test_df[required_cols], processed_data["test"])


def pre_process(text: str, rm_stops: bool = False) -> str:
    """Clean the text message

    Parameters
    ----------
    text : str
        original text
    rm_stops : bool, optional
        if stopwords need to be removed, by default False

    Returns
    -------
    str
        processed text with only Chinese characters
    """
    if rm_stops:
        text = rm_stopwords(text)

    frags = [frag for frag in re.findall(r"[\u4e00-\u9fff]+", text)]

    return "".join(frags) if len(frags) > 0 else text


def rm_stopwords(text: str) -> str:
    """Remove stopwords from the text

    Parameters
    ----------
    text : str
        original text

    Returns
    -------
    str
        processed text with no stopword
    """
    return "".join([char for char in text if char not in ZH_STOPS])


def write_processed_data(df: pd.DataFrame, data_dir: str):
    """Write processed data to local disk

    Parameters
    ----------
    df : pd.DataFrame
        processed comment data

        Required fields:
            - processed_text
            - label

    data_dir : str
        local path for exporting the data
    """
    df.to_csv(data_dir, index=False)


if __name__ == "__main__":
    main()
