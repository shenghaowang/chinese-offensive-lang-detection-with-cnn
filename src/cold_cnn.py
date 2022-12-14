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
        cross_entropy_loss = F.binary_cross_entropy(y_hat, y, reduction="mean")

        # Set constraint on the model params of
        # the last fully connected layer
        l2_lambda = 0.01
        fc_params = torch.cat([x.view(-1) for x in self.model.fc2.parameters()])
        loss = cross_entropy_loss + l2_lambda * torch.norm(fc_params, 2)

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

        self.kernel_sizes = hyparams.kernel_sizes

        agg_seq_len = 0
        for kernel_size in self.kernel_sizes:
            module_name = f"conv-maxpool-{kernel_size}"

            conv_seq_len = self.compute_seq_len(
                seq_len, kernel_size, hyparams.cnn_stride
            )
            pooling_seq_len = self.compute_seq_len(
                conv_seq_len, kernel_size, hyparams.pooling_stride
            )
            module = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hyparams.out_channels,
                    kernel_size=kernel_size,
                    stride=hyparams.cnn_stride,
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(
                    kernel_size=kernel_size, stride=hyparams.pooling_stride
                ),
            )

            agg_seq_len += pooling_seq_len
            setattr(self, module_name, module)

        self.flatten = torch.nn.Flatten()
        self.dropout1 = torch.nn.Dropout(p=hyparams.dropouts.p1)
        self.fc1 = torch.nn.Linear(
            hyparams.out_channels * agg_seq_len, hyparams.fc_features
        )
        self.dropout2 = torch.nn.Dropout(p=hyparams.dropouts.p2)
        self.fc2 = torch.nn.Linear(hyparams.fc_features, 1)
        self.activation = torch.nn.Sigmoid()

    @staticmethod
    def compute_seq_len(input_height: int, kernel_size: int, stride: int) -> int:
        return int((input_height - kernel_size) / stride) + 1

    def forward(self, batch):

        conv_maxpool_outputs = [
            getattr(self, f"conv-maxpool-{kernel_size}")(batch)
            for kernel_size in self.kernel_sizes
        ]
        x = torch.cat(conv_maxpool_outputs, axis=2)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return self.activation(x)
