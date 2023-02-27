# Spacecutter Lightning

A PyTorch Lightning Callback for Spacecutter.

## Installation

```bash
pip install spacecutter-lightning
```

## Usage

```python
import torch
import pytorch_lightning as pl
from spacecutter_lightning import ClipCutpoints
from spacecutter import OrdinalLogisticHead, CumulativeLinkLoss

num_classes = 10
num_features = 5
hidden_size = 10
size = 200

x = torch.randn(size, num_features)
y = torch.randint(0, num_classes, (size, 1))

train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y),
)

model = torch.nn.Sequential(
    torch.nn.Linear(num_features, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
    OrdinalLogisticHead(num_classes),
)

loss_fn = CumulativeLinkLoss()


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


trainer = pl.Trainer(
    callbacks=[ClipCutpoints()],
    max_epochs=10,
)
trainer.fit(LitModel(), train_dataloader)
```