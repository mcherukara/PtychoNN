import glob
import logging
import os
import pathlib

import click
import numpy as np
import numpy.typing as npt
import torch
import torchinfo

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
        out_dir=out_dir,
        epochs=epochs,
        batch_size=32,
    )


def train(
    X_train: npt.NDArray[float],
    Y_train: npt.NDArray[float],
    out_dir: pathlib.Path | None,
    load_model_path: pathlib.Path | None = None,
    epochs: int = 1,
    batch_size: int = 32,
):
    """Train a PtychoNN model.

    Parameters
    ----------
    X_train (N, WIDTH, HEIGHT)
        The diffraction patterns.
    Y_train (N, 2, WIDTH, HEIGHT)
        The corresponding reconstructed patches for the diffraction patterns.
    out_dir
        A folder where all the training artifacts are saved.
    load_model_path
        Load a previous model's parameters from this file.
    """
    assert Y_train.dtype == np.float32
    assert np.all(np.isfinite(Y_train))
    assert X_train.dtype == np.float32
    assert np.all(np.isfinite(X_train))
    assert X_train.ndim == 4
    assert Y_train.ndim == 4

    traindata = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train),
    )

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    trainer = lightning.Trainer(
        max_epochs=epochs,
        default_root_dir=out_dir,
    )

    model = ptychonn.model.LitReconSmallModel()

    if load_model_path is not None:
        model = ptychonn.model.LitReconSmallModel.load_from_checkpoint(load_model_path)

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
    )

    return trainer
