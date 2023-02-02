#!/usr/bin/env python
"""Tests for `ptychonn` package."""
import pathlib

from click.testing import CliRunner
import torch

import ptychonn.model
from ptychonn import __main__ as cli

_test_dir = pathlib.Path(__file__).resolve().parents[0]

def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'ptychonn.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Show this message and exit.' in help_result.output


def test_load_weights():
    model = ptychonn.model.ReconSmallPhaseModel(16, False)
    model.load_state_dict(torch.load(_test_dir / 'weights.pth'))
    print(model)
