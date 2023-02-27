from typing import List

import pytest
import torch.utils.data
from torch import nn
import pytorch_lightning as pl
from spacecutter_lightning import ClipCutpoints
from spacecutter.models import OrdinalLogisticHead
from spacecutter.losses import CumulativeLinkLoss


class LitModel(pl.LightningModule):
    def __init__(self, num_classes: int = 10, num_features: int = 10, hidden_size: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            OrdinalLogisticHead(num_classes=num_classes),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = CumulativeLinkLoss(reduction="sum")(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def is_ascending(values: List[float]) -> bool:
    """
    Check if the values are in ascending order.

    Parameters
    ----------
    values: List[float]
        The values to check.

    Returns
    -------
    is_ascending: bool
        True if the values are in ascending order, False otherwise.
    """
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


@pytest.mark.filterwarnings("ignore:The dataloader, train_dataloader, does not have many workers:UserWarning")
@pytest.mark.filterwarnings("ignore:.* available but not used. Set `accelerator` and `devices`:UserWarning")
def test_pl_callback():
    pl.seed_everything(42)
    num_classes = 5
    num_features = 5
    size = 200
    X = torch.rand(size, num_features)
    y = torch.randint(0, num_classes, (size, 1)).long()
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y))
    model = LitModel(
        num_classes=num_classes,
        num_features=num_features,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=[ClipCutpoints()],
        max_epochs=10,
        logger=[],  # disable logging
        enable_checkpointing=False,  # disable checkpointing
    )

    trainer.fit(model, dataloader)
    head: OrdinalLogisticHead = model.model[-1]
    assert is_ascending(head.link.cutpoints.tolist())
