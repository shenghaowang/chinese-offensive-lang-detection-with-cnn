from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import zh_core_web_md

# from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def load_data(data_file: str, X_col: str, y_col: str) -> List[Tuple[str, int]]:
    """Load comments and labels

    Parameters
    ----------
    data_file : str
        directory of the data file
    X_col : str
        name of the comment column
    y_col : str
        name of the label column

    Returns
    -------
    List[Tuple[str, int]]
        list of loaded comments and labels
    """
    df = pd.read_csv(data_file)
    X = df[X_col].to_numpy()
    y = df[y_col].to_numpy()

    data = []
    for idx, comment in enumerate(X):
        data.append((comment, y[idx]))

    return data


class ColdVectorizer:
    def __init__(self):
        """Create word vectors from given comments"""
        self.model = zh_core_web_md.load()

    def vectorize(self, words):
        """
        Given a sentence, tokenize it and returns a list of
        pre-trained word vector for each token.
        """

        word_vecs = []

        # Split on words
        for _, word in enumerate(words.split()):
            # Tokenize the words using spacy
            spacy_doc = self.model.make_doc(word)
            vec = [token.vector for token in spacy_doc]
            word_vecs.append(vec)

        return word_vecs


class ColdDataset(Dataset):
    """Creates an pytorch dataset to consume our pre-loaded text data
    Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(self, data, vectorizer):
        self.dataset = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        comment, label = self.dataset[idx]
        word_vecs = self.vectorizer.vectorize(comment)

        return {
            "vectors": word_vecs,
            "label": label,
            "comment": comment,  # for debugging only
        }


class ColdDataModule(pl.LightningDataModule):
    """LightningDataModule: Wrapper class for the dataset to be used in training"""

    def __init__(
        self,
        vectorizer,
        batch_size,
        max_seq_len,
        train_data: List[Tuple],
        valid_data: List[Tuple],
        test_data: List[Tuple],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.cold_train = ColdDataset(train_data, vectorizer)
        self.cold_valid = ColdDataset(valid_data, vectorizer)
        self.cold_test = ColdDataset(test_data, vectorizer)

    def collate_fn(self, batch):
        """Convert the input raw data from the dataset into model input"""
        # Sort batch according to sequence length
        # This is for "pack_padded_sequence" in LSTM
        # Order speeds it up.
        batch.sort(key=lambda x: len(x["vectors"]), reverse=True)

        # Separate out the vectors and labels from the batch
        word_vector = [
            torch.Tensor(np.squeeze(item["vectors"], axis=0)) for item in batch
        ]

        # Trim sequences to ensure consistent length
        word_vector = [
            torch.nn.ZeroPad2d((0, 0, 0, self.max_seq_len - len(vec)))(vec)
            if self.max_seq_len > len(vec)
            else vec[: self.max_seq_len, :]
            for vec in word_vector
        ]

        labels = torch.LongTensor(np.array([item["label"] for item in batch]))

        # Pad each vector sequence to the same size
        # [batch_size, word_vec_dim, sequence_length]
        padded_word_vector = pad_sequence(word_vector, batch_first=True).transpose(1, 2)

        return {
            "vectors": padded_word_vector,
            "label": labels,
            "comments": [item["comment"] for item in batch],
        }

    def train_dataloader(self):
        return DataLoader(
            self.cold_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cold_valid,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cold_test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
