import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig


class OffensiveLangDetector(pl.LightningModule):
    def __init__(self, model, learning_rate: float):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.f1 = torchmetrics.F1Score(num_classes=2)

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform labels to one-hot vectors
        x = batch["vectors"]
        y = batch["labels"]

        # Perform prediction and calculate loss and F1 score
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y, reduction="mean")

        # Logging
        self.log_dict(
            {
                f"{mode}_loss": loss,
                f"{mode}_f1": self.f1(predictions, y),
            },
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self._calculate_loss(batch, "val")
        return loss

    def test_step(self, batch, batch_nb):
        loss = self._calculate_loss(batch, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["vectors"]
        x_len = batch["vectors_length"]
        y_hat = self.model(x, x_len)
        predictions = torch.argmax(y_hat, dim=1)

        return {
            "logits": y_hat,
            "predictions": predictions,
            "labels": batch["labels"],
            "comments": batch["comments"],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_epoch_start(self):
        """Create a new progress bar for each epoch"""
        print("\n")


class ColdCNN(torch.nn.Module):
    def __init__(self, hyparams: DictConfig, seq_len: int):
        super(ColdCNN, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=hyparams.in_channels,
            out_channels=hyparams.out_channels,
            kernel_size=(hyparams.kernel_height, hyparams.kernel_width),
            stride=hyparams.cnn_stride,
        )
        seq_len = self.compute_seq_len(
            seq_len, hyparams.kernel_height, hyparams.cnn_stride
        )

        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(
            kernel_size=(hyparams.kernel_height, 1), stride=hyparams.pooling_stride
        )
        seq_len = self.compute_seq_len(
            seq_len, hyparams.kernel_height, hyparams.pooling_stride
        )

        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(hyparams.out_channels * seq_len, 1)
        self.activation = torch.nn.Sigmoid()

    @staticmethod
    def compute_seq_len(input_height: int, kernel_height: int, stride: int) -> int:
        return int((input_height - kernel_height) / stride) + 1

    def forward(self, batch):

        x = self.conv(batch)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.activation(self.fc(x))

        return x
