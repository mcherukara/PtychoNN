import numpy as np
import torch
import torch.nn as nn


class ReconSmallPhaseModel(nn.Module):

    def __init__(self, nconv: int = 16, use_batch_norm=False):
        super(ReconSmallPhaseModel, self).__init__()
        self.nconv = nconv
        self.use_batch_norm = use_batch_norm

        self.encoder = nn.Sequential(  # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8),
        )

        # amplitude model
        #self.decoder_amplitude = nn.Sequential(
        #    *self.up_block(self.nconv * 8, self.nconv * 8),
        #    *self.up_block(self.nconv * 8, self.nconv * 4),
        #    *self.up_block(self.nconv * 4, self.nconv * 2),
        #    *self.up_block(self.nconv * 2, self.nconv * 1),
        #    nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
        #)

        # phase model
        self.decoder_phase = nn.Sequential(
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(1) if self.use_batch_norm else torch.nn.Identity(),
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
            ),
            nn.BatchNorm2d(filters_out)
            if self.use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out)
            if self.use_batch_norm else torch.nn.Identity(),
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
            ),
            nn.BatchNorm2d(filters_out)
            if self.use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out)
            if self.use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            #amplitude = self.decoder_amplitude(x1)
            phase = self.decoder_phase(x1)

            #Restore -pi to pi range
            phase = phase * np.pi  #Using tanh activation (-1 to 1) for phase so multiply by pi

        return phase
