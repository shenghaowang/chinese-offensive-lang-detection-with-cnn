import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from stopwordsiso import stopwords

import hydra

ZH_STOPS = stopwords("zh")


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
    df["len"] = df["TEXT"].apply(len)
    logger.info(f"Length of comments:\n{df['len'].describe()}")
    logger.info(
        f"95th percentile of the comment length: {np.percentile(df['len'].values, 95)}"
    )

    # Preview text processing
    logger.debug("Preview the processed text:")
    for text in df["TEXT"][:5]:
        logger.debug(f"Original text: {text}")
        logger.debug(f"Processed text: {pre_process(text, rm_stops)}")
        logger.debug("\n")

    # Create directory for storing the processed data
    output_dir = Path(processed_data.train).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_type in ["train", "dev", "test"]:
        logger.info(f"Process {ds_type} ...")
        df = pd.read_csv(raw_data[ds_type])
        logger.info(f"{len(df)} raw comments loaded from: {raw_data[ds_type]}")

        # Clean the comments
        df["processed_text"] = df["TEXT"].apply(pre_process)

        df = df[df["processed_text"].apply(lambda x: len(x.strip()) > 0)]
        logger.info(f"{len(df)} comments remain after text cleaning.")

        df[["processed_text", "label"]].to_csv(processed_data[ds_type], index=False)
        logger.info(f"Processed reviews written to {processed_data[ds_type]}")


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


if __name__ == "__main__":
    main()
