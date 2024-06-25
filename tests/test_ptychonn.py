#!/usr/bin/env python
"""Tests for `ptychonn` package."""
import pathlib
import pprint

from click.testing import CliRunner
import ptychonn
import numpy as np
import matplotlib.pyplot as plt

from ptychonn import __main__ as cli

_test_dir = pathlib.Path(__file__).resolve().parents[0]


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Show this message and exit.' in help_result.output


def test_load_weights():
    model = ptychonn.ReconSmallModel()
    trainer = ptychonn.Trainer(
        model=model,
        batch_size=32,
    )
    trainer.updateSavedModel(
        trainer.model,
        _test_dir,
        suffix='',
    )
    tester = ptychonn.Tester(
        model=model,
        model_params_path=_test_dir / 'best_model.pth',
    )

def test_train_simulated():

    with np.load('simulated_data.npz') as f:
        data = f['reciprocal']
        patches = f['real']

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

    trainer, logger = ptychonn.train(
        X_train=data,
        Y_train=patches,
        model=ptychonn.init_or_load_model(
            ptychonn.LitReconSmallModel,
            model_checkpoint_path=None,
            model_init_params=dict(),
        ),
        out_dir=None,
        epochs=500,
        batch_size=32,
        devices=-1,
        strategy='ddp_spawn',
    )

    ptychonn.create_model_checkpoint(trainer, 'trained-model.checkpoint')

    with open("trained-model.log", 'w') as log_file:
        pprint.pprint(logger.logs, log_file, indent=4,)

def test_infer_simulated():

    with np.load('simulated_test.npz') as f:
        data = f['reciprocal']
        scan = f['scan']

    inferences = ptychonn.infer(
        data=data,
        model=ptychonn.init_or_load_model(
            ptychonn.model.LitReconSmallModel,
            model_checkpoint_path='trained-model.checkpoint',
            model_init_params=None,
        )
    )

    # Plotting some summary images

    pstitched = ptychonn.stitch_from_inference(
        inferences[:, 0],
        scan,
        stitched_pixel_width=1,
        inference_pixel_width=1,
    )
    plt.figure(1, figsize=[8.5, 7])
    plt.imshow(pstitched)
    plt.colorbar()
    plt.tight_layout()
    plt.title('stitched_phases')
    plt.savefig('pstitched.png', bbox_inches='tight')

    astitched = ptychonn.stitch_from_inference(
        inferences[:, 1],
        scan,
        stitched_pixel_width=1,
        inference_pixel_width=1,
    )
    plt.figure(2, figsize=[8.5, 7])
    plt.imshow(astitched)
    plt.colorbar()
    plt.tight_layout()
    plt.title('stitched_amplitudes')
    plt.savefig('astitched.png', bbox_inches='tight')

    test_inferences = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 4, figsize=[13, 3])
    for ix, inf in enumerate(test_inferences):
        plt.subplot(2, 4, ix + 1)
        plt.pcolormesh(inferences[inf, 0])
        plt.colorbar()
        plt.title('Inference at position {0}'.format(inf))
        plt.subplot(2, 4, 4 + ix + 1)
        plt.pcolormesh(inferences[inf, 1])
        plt.colorbar()
        plt.title('Inference at position {0}'.format(inf))
    plt.tight_layout()
    plt.savefig('inferences.png', bbox_inches='tight')

    return 0

if __name__ == '__main__':
    test_train_simulated()
    test_infer_simulated()
