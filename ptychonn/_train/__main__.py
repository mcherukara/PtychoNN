import argparse
import glob
import logging
import os
import pathlib
import typing

import click
import lightning
import numpy as np
import numpy.typing as npt
import torch

import ptychonn.model
import ptychonn.plot

logger = logging.getLogger(__name__)


@click.command(name='train')
@click.argument(
    'data_dir',
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'out_dir',
    type=click.Path(
        exists=False,
        writable=True,
        dir_okay=True,
        file_okay=False,
        path_type=pathlib.Path,
    ),
)
@click.option(
    '--epochs',
    type=click.INT,
    default=100,
)
def train_cli(
    data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    epochs: int,
):
    """Train a model from diffraction patterns and reconstructed patches.

    Expects a `DATA_DIR` of NPZ files. Each contains two parameters:
    `reciprocal` and `real` which are the diffraction patterns and
    reconstructed (complex-valued) patches. The shapes of these two arrays
    match. The data from every file in the folder will be concatinated into one
    training/validation set. All the training artifacts are saved into
    `OUT_DIR`.
    """
    logging.basicConfig(level=logging.INFO, )

    dataslist = []
    patchlist = []

    for name in glob.glob(str(data_dir / '*.npz')):
        print(name)
        with np.load(name) as f:
            dataslist.append(f['reciprocal'])
            patchlist.append(f['real'])

    data = np.concatenate(dataslist, axis=0)
    patches = np.concatenate(patchlist, axis=0)

    # NOTE: @carterbox AFAIK, the only preprocessing of the training data is a
    # centering of the phase in the center 3rd of the reconstructed patches.
    # The diffraction patterns are converted to float32 and otherwise
    # unaltered.
    phase = np.angle(patches).astype('float32')
    phase -= np.mean(
        phase[..., phase.shape[-2] // 3:-phase.shape[-2] // 3,
              phase.shape[-1] // 3:-phase.shape[-1] // 3], )
    amplitude = np.abs(patches).astype('float32')
    patches = np.stack((phase, amplitude), axis=1)

    os.makedirs(out_dir, exist_ok=True)

    train(
        X_train=data,
        Y_train=patches,
        model=init_or_load_model(
            ptychonn.LitReconSmallModel,
            model_checkpoint_path=None,
            model_init_params=dict(),
        ),
        out_dir=out_dir,
        epochs=epochs,
        batch_size=32,
    )


class ListLogger(lightning.pytorch.loggers.logger.Logger):
    """An in-memory logger that saves logged parameters to a List

    Parameters
    ----------
    logs :
        Each entry of this list is a dictionary with parameter name value
        pairs. Each entry of the list represents the parameters during a single
        step.
    hyperparameters :
        Some hyperparameters that were logged?

    """
    def __init__(self):
        super().__init__()
        self.logs: typing.List[typing.Dict] = []
        self.hyperparameters: argparse.Namespace = argparse.Namespace()

    @lightning.pytorch.utilities.rank_zero_only
    def log_metrics(self, metrics, step=None):
        metrics["step"] = step
        self.logs.append(metrics)

    @lightning.pytorch.utilities.rank_zero_only
    def log_hyperparams(self, params):
        self.hyperparameters = params

    @lightning.pytorch.utilities.rank_zero_only
    def save(self):
        # No need to save anything for this logger
        pass

    @lightning.pytorch.utilities.rank_zero_only
    def finalize(self, status):
        # Finalize the logger
        pass

    @property
    def name(self):
        return "ListLogger"

    @property
    def version(self):
        return "0.1.0"


def train(
    X_train: npt.NDArray[np.float32],
    Y_train: npt.NDArray[np.float32],
    model: lightning.LightningModule,
    out_dir: pathlib.Path | None,
    epochs: int = 1,
    batch_size: int = 32,
) -> typing.Tuple[lightning.Trainer, lightning.pytorch.loggers.CSVLogger | ListLogger]:
    """Train a PtychoNN model.

    Initialize a model for the model parameter using the `init_or_load_model()`
    function.

    If out_dir is not None the following artifacts will be created:
        - {out_dir}/best_model.ckpt
        - {out_dir}/metrics.csv
        - {out_dir}/hparams.yaml
        - {out_dir}/metrics.png

    Parameters
    ----------
    X_train : (N, WIDTH, HEIGHT)
        The diffraction patterns.
    Y_train : (N, 2, WIDTH, HEIGHT)
        The corresponding reconstructed patches for the diffraction patterns.
    out_dir
        A folder where all the training artifacts are saved.
    model
        An initialized PtychoNN model.
    epochs
        The maximum number of training epochs
    batch_size
        The size of one training batch.
    """
    if out_dir is not None:
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=out_dir,
            filename="best_model",
            save_top_k=1,
            monitor="training_loss",
            mode="min",
        )

        logger = lightning.pytorch.loggers.CSVLogger(
            save_dir=out_dir,
            name="",
            version="",
            prefix="",
        )

    else:
        logger = ListLogger()

    trainer = lightning.Trainer(
        max_epochs=epochs,
        default_root_dir=out_dir,
        callbacks=None if out_dir is None else [checkpoint_callback],
        logger=logger,
        enable_checkpointing=False if out_dir is None else True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=create_training_dataloader(
            X_train,
            Y_train,
            batch_size,
        ),
    )

    if out_dir is not None:
        with open(out_dir / "metrics.csv") as f:
            headers = f.readline().strip("\n").split(",")
        numbers = np.genfromtxt(
            out_dir / "metrics.csv",
            delimiter=",",
            skip_header=1,
        )
        metrics = dict()
        for col, header in enumerate(headers):
            metrics[header] = numbers[:, col]

        ptychonn.plot.plot_metrics(
            metrics=metrics,
            save_fname=out_dir / "metrics.png",
        )

    return trainer, logger


def create_training_dataloader(
    X_train: npt.NDArray[np.float32],
    Y_train: npt.NDArray[np.float32],
    batch_size: int = 32,
) -> torch.utils.data.DataLoader:
    """Create a Pytorch Dataloader from NumPy arrays."""

    assert Y_train.dtype == np.float32
    assert np.all(np.isfinite(Y_train))
    assert X_train.dtype == np.float32
    assert np.all(np.isfinite(X_train))

    if X_train.ndim != 3:
        msg = (
            "X_train must have 3 dimemnsions: (N, WIDTH, HEIGHT); "
            f" not {X_train.shape}"
        )
        raise ValueError(msg)
    if Y_train.ndim != 4:
        msg = (
            f"Y_train must have 4 dimensions: (N, [1,2], WIDTH, HEIGHT); "
            f"not {Y_train.shape}"
        )
        raise ValueError(msg)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train[:, None, :, :]),
        torch.from_numpy(Y_train),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader


def init_or_load_model(
    model_type: typing.Type[lightning.LightningModule],
    *,
    model_checkpoint_path: pathlib.Path | None,
    model_init_params: dict | None,
):
    """Initialize one of the PtychoNN models via params or a checkpoint."""
    if not (model_checkpoint_path is None or model_init_params is None):
        msg = (
            "One of model_checkpoint_path OR model_init_params must be None! "
            "Both cannot be defined."
        )
        raise ValueError(msg)

    if model_checkpoint_path is not None:
        return model_type.load_from_checkpoint(model_checkpoint_path)
    else:
        return model_type(**model_init_params)
