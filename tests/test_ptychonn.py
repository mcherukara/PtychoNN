#!/usr/bin/env python
"""Tests for `ptychonn` package."""
import pathlib

from click.testing import CliRunner
import ptychonn

import ptychonn.model
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
