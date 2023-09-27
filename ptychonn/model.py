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

    Shapes
    ------
    input : (N, 1, H, W)
        The measured intensity of the diffraction patterns
    output : (N, C, H, W)
        The phase (and amplitude if C is 2) in the patch of the object
    """

    def __init__(
        self,
        nconv: int = 16,
        use_batch_norm: bool = True,
        enable_amplitude: bool = True,
    ):
        super().__init__()
        self.nconv = nconv
        self.use_batch_norm = use_batch_norm
        self.enable_amplitude = enable_amplitude

        # Appears sequential has similar functionality as TF avoiding need for
        # separate model definition and activ
        self.encoder = nn.Sequential(
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8),
        )

        # Double the number of channels when doing both phase and amplitude,
        # but keep them separate with grouping
        c = 2 if self.enable_amplitude else 1
        self.decoder = nn.Sequential(
            *self.up_block(self.nconv * 8 * 1, self.nconv * 8 * c, groups=c),
            *self.up_block(self.nconv * 8 * c, self.nconv * 4 * c, groups=c),
            *self.up_block(self.nconv * 4 * c, self.nconv * 2 * c, groups=c),
            *self.up_block(self.nconv * 2 * c, self.nconv * 1 * c, groups=c),
            nn.Conv2d(
                in_channels=self.nconv * 1 * c,
                out_channels=c,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
                groups=c,
            ),
            *((nn.BatchNorm2d(c), ) if self.use_batch_norm else ()),
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
                in_channels=filters_out,
                out_channels=filters_out,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
            ), *((nn.BatchNorm2d(filters_out), ) if self.use_batch_norm else
                 ()),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

    def up_block(self, filters_in: int, filters_out: int, groups: int):
        return [
            nn.Conv2d(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
                groups=groups,
            ), *((nn.BatchNorm2d(filters_out), ) if self.use_batch_norm else
                 ()),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=filters_out,
                out_channels=filters_out,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                bias=(not self.use_batch_norm),
                groups=groups,
            ), *((nn.BatchNorm2d(filters_out), ) if self.use_batch_norm else
                 ()),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]

    def forward(self, x):
        with torch.cuda.amp.autocast():
            output = self.decoder(self.encoder(x))
            # Restore -pi to pi range
            # Using tanh activation (-1 to 1) for phase so multiply by pi
            output[..., 0, :, :] = torch.tanh(output[..., 0, :, :]) * np.pi
            # Restrict amplitude to (0, 1) range with sigmoid
            if self.enable_amplitude:
                output[..., 1, :, :] = torch.sigmoid(output[..., 1, :, :])
        return output
