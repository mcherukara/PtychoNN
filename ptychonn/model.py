"""Define PtychoNN Pytorch models."""

import numpy as np
import torch
import torch.nn as nn


class ReconSmallModel(nn.Module):
    """A small PychoNN model.

    Parameters
    ----------
    nconv :
        The number of convolution kernels at the smallest level
    use_batch_norm :
        Whether to use batch normalization after convolution layers
    enable_phase :
        Whether the phase branch is included in the model
    enable_amplitude :
        Whether the amplitude branch is included in the model

    """

    def __init__(
        self,
        nconv: int = 16,
        use_batch_norm: bool = False,
        enable_phase: bool = True,
        enable_amplitude: bool = True,
    ):
        super(ReconSmallPhaseModel, self).__init__()
        self.nconv = nconv
        self.use_batch_norm = use_batch_norm
        self.enable_phase = enable_phase
        self.enable_amplitude = enable_amplitude
        if not (enable_phase or enable_amplitude):
            msg = 'This module must reconstruct at least one of phase or amplitude.'
            raise ValueError(msg)

        # Appears sequential has similar functionality as TF avoiding need for
        # separate model definition and activ
        self.encoder = nn.Sequential(
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8),
        )

        # amplitude model
        if enable_amplitude:
            self.decoder1 = nn.Sequential(
                *self.up_block(self.nconv * 8, self.nconv * 8),
                *self.up_block(self.nconv * 8, self.nconv * 4),
                *self.up_block(self.nconv * 4, self.nconv * 2),
                *self.up_block(self.nconv * 2, self.nconv * 1),
                nn.Conv2d(
                    self.nconv * 1,
                    1,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=(not self.use_batch_norm),
                ),
                *((nn.BatchNorm2d(1), ) if self.use_batch_norm else ()),
                nn.Sigmoid(),
            )

        # phase model
        if enable_phase:
            self.decoder2 = nn.Sequential(
                *self.up_block(self.nconv * 8, self.nconv * 8),
                *self.up_block(self.nconv * 8, self.nconv * 4),
                *self.up_block(self.nconv * 4, self.nconv * 2),
                *self.up_block(self.nconv * 2, self.nconv * 1),
                nn.Conv2d(
                    self.nconv * 1,
                    1,
                    3,
                    stride=1,
                    padding=(1, 1),
                    bias=(not self.use_batch_norm),
                ),
                *((nn.BatchNorm2d(1), ) if self.use_batch_norm else ()),
                nn.Tanh(),
            )

    def down_block(self, filters_in, filters_out):
        return [
            nn.Conv2d(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
            ), *((nn.BatchNorm2d(filters_out), ) if self.use_batch_norm else
                 ()),
            nn.ReLU(),
            nn.Conv2d(
                filters_out,
                filters_out,
                3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
            ), *((nn.BatchNorm2d(filters_out), ) if self.use_batch_norm else
                 ()),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

    def up_block(self, filters_in, filters_out):
        return [
            nn.Conv2d(
                filters_in,
                filters_out,
                3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
            ), *((nn.BatchNorm2d(filters_out), ) if self.use_batch_norm else
                 ()),
            nn.ReLU(),
            nn.Conv2d(
                filters_out,
                filters_out,
                3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
            ), *((nn.BatchNorm2d(filters_out), ) if self.use_batch_norm else
                 ()),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)

            if self.enable_amplitude:
                amplitude = self.decoder1(x1)

            if self.enable_phase:
                phase = self.decoder2(x1)

                # Restore -pi to pi range
                # Using tanh activation (-1 to 1) for phase so multiply by pi
                phase = phase * np.pi

        if self.enable_amplitude and self.enable_phase:
            return amplitude, phase

        if self.enable_amplitude:
            return amplitude

        if self.enable_phase:
            return phase


class ReconSmallPhaseModel(ReconSmallModel):
    """Alias for a phase-only ReconSmallModel."""

    def __init__(
        self,
        nconv: int = 16,
        use_batch_norm=False,
    ) -> None:
        super().__init__(
            nconv=nconv,
            use_batch_norm=use_batch_norm,
            enable_phase=True,
            enable_amplitude=False,
        )


class ReconSmallAmpliModel(ReconSmallModel):
    """Alias for an amplitude-only ReconSmallModel."""

    def __init__(
        self,
        nconv: int = 16,
        use_batch_norm=False,
    ) -> None:
        super().__init__(
            nconv=nconv,
            use_batch_norm=use_batch_norm,
            enable_amplitude=True,
            enable_phase=False,
        )
