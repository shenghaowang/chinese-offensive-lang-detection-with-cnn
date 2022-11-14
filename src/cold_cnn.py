import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


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
    def __init__(self):
        super(ColdCNN, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=(3, 300), stride=1
        )
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.fc = torch.nn.Linear(1)
        self.sigmoid = torch.sigmoid()

    def forward(self, batch, batch_len):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            batch, batch_len, batch_first=True, enforce_sorted=True
        )
        x = self.conv(packed_input)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.sigmoid(self.fc(x))

        return x
