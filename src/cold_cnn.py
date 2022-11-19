import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

# from loguru import logger
from omegaconf import DictConfig


class OffensiveLangDetector(pl.LightningModule):
    def __init__(self, model, learning_rate: float):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.f1 = torchmetrics.F1Score(multiclass=False)

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform labels
        x = batch["vectors"]
        y = batch["label"].to(torch.float32)

        # Perform prediction and calculate loss and F1 score
        y_hat = torch.squeeze(self(x))
        predictions = y_hat.round()
        loss = F.binary_cross_entropy(y_hat, y, reduction="mean")

        # Logging
        self.log_dict(
            {
                f"{mode}_loss": loss,
                f"{mode}_f1": self.f1(predictions, batch["label"]),
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
        # x_len = batch["vectors_length"]
        y_hat = self.model(x)
        predictions = torch.argmax(y_hat, dim=1)

        return {
            "logits": y_hat,
            "predictions": predictions,
            "label": batch["label"],
            "comments": batch["comments"],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_epoch_start(self):
        """Create a new progress bar for each epoch"""
        print("\n")


class ColdCNN(torch.nn.Module):
    def __init__(self, hyparams: DictConfig, in_channels: int, seq_len: int):
        super(ColdCNN, self).__init__()

        self.kernel_heights = hyparams.kernel_heights

        agg_seq_len = 0
        for kernel_height in self.kernel_heights:
            module_name = f"conv-maxpool-{kernel_height}"

            conv_seq_len = self.compute_seq_len(
                seq_len, kernel_height, hyparams.cnn_stride
            )
            pooling_seq_len = self.compute_seq_len(
                conv_seq_len, kernel_height, hyparams.pooling_stride
            )
            module = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hyparams.out_channels,
                    kernel_size=kernel_height,
                    stride=hyparams.cnn_stride,
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(
                    kernel_size=kernel_height, stride=hyparams.pooling_stride
                ),
            )

            agg_seq_len += pooling_seq_len
            setattr(self, module_name, module)

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(p=hyparams.dropout)
        self.fc1 = torch.nn.Linear(
            hyparams.out_channels * agg_seq_len, hyparams.fc_features
        )
        self.fc2 = torch.nn.Linear(hyparams.fc_features, 1)
        self.activation = torch.nn.Sigmoid()

    @staticmethod
    def compute_seq_len(input_height: int, kernel_height: int, stride: int) -> int:
        return int((input_height - kernel_height) / stride) + 1

    def forward(self, batch):

        conv_maxpool_outputs = [
            getattr(self, f"conv-maxpool-{kernel_height}")(batch)
            for kernel_height in self.kernel_heights
        ]
        x = torch.cat(conv_maxpool_outputs, axis=2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return self.activation(x)
