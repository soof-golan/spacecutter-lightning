import pytorch_lightning as pl
import torch
from spacecutter.callbacks import AscensionCallback


class ClipCutpoints(pl.Callback):
    """
    Ensure that each cutpoint is ordered in ascending value.
    e.g.

    .. < cutpoint[i - 1] < cutpoint[i] < cutpoint[i + 1] < ...

    This is done by clipping the cutpoint values at the end of a batch gradient
    update. By no means is this an efficient way to do things, but it works out
    of the box with stochastic gradient descent.

    Parameters
    ----------
    margin : float, (default=0.0)
        The minimum value between any two adjacent cutpoints.
        e.g. enforce that cutpoint[i - 1] + margin < cutpoint[i]
    min_val : float, (default=-1e6)
        Minimum value that the smallest cutpoint may take.
    """

    def __init__(self, margin: float = 0.0, min_val: float = -1.0e6) -> None:
        super().__init__()
        self.ascension_callback = AscensionCallback(margin=margin, min_val=min_val)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        with torch.no_grad():
            pl_module.apply(self.ascension_callback)
