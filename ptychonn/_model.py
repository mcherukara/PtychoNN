import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class ReconSmallPhaseModel(nn.Module):
    def __init__(self, nconv: int = 16):
        super(ReconSmallPhaseModel, self).__init__()
        self.nconv = nconv

        self.encoder = nn.Sequential(  # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8))

        # amplitude model
        #self.decoder1 = nn.Sequential(
        #    *self.up_block(self.nconv * 8, self.nconv * 8),
        #    *self.up_block(self.nconv * 8, self.nconv * 4),
        #    *self.up_block(self.nconv * 4, self.nconv * 2),
        #    *self.up_block(self.nconv * 2, self.nconv * 1),
        #    nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
        #)

        # phase model
        self.decoder2 = nn.Sequential(
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh())

    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(in_channels=filters_in,
                      out_channels=filters_out,
                      kernel_size=3,
                      stride=1,
                      padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]
        return block

    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]
        return block

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            #amp = self.decoder1(x1)
            ph = self.decoder2(x1)

            #Restore -pi to pi range
            ph = ph * np.pi  #Using tanh activation (-1 to 1) for phase so multiply by pi

        return ph


class Tester():
    def __init__(
        self,
        model: ReconSmallPhaseModel,
        batch_size: int,
        model_params_path: str,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        self.model = model
        self.batch_size = batch_size
        self.model_params_path = model_params_path

        self.model.load_state_dict(
            torch.load(self.model_params_path, map_location=self.device))
        self.model = nn.DataParallel(self.model)  #Default all devices
        self.model = self.model.to(self.device)

    def setTestData(self, X_test: np.ndarray):
        self.X_test = torch.tensor(X_test[:, None, ...].astype('float32'))
        self.test_data = TensorDataset(self.X_test)

        self.testloader = DataLoader(self.test_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=4)

    def predictTestData(self, npz_save_path: str = None):
        self.model.eval()
        phs_eval = []
        for i, ft_images in enumerate(self.testloader):
            ft_images = ft_images[0].to(self.device)
            ph_eval = self.model(ft_images)
            for j in range(ft_images.shape[0]):
                phs_eval.append(ph_eval[j].detach().to("cpu").numpy())
        self.phs_eval = np.array(phs_eval).squeeze().astype('float32')
        if npz_save_path is not None:
            np.savez_compressed(npz_save_path, ph=self.phs_eval)
            print(f'Finished the inference stage and saved at {npz_save_path}')
        return self.phs_eval

    def calcErrors(self, phs_true: np.ndarray, npz_save_path: str = None):
        from skimage.metrics import mean_squared_error as mse

        self.phs_true = phs_true
        self.errors = []
        for i, (p1, p2) in enumerate(zip(self.phs_eval, self.phs_true)):
            err2 = mse(p1, p2)
            self.errors.append([err2])

        self.errors = np.array(self.errors)
        print("Mean errors in phase")
        print(np.mean(self.errors, axis=0))

        if npz_save_path is not None:
            np.savez_compressed(npz_save_path, phs_err=self.errors[:, 0])

        return self.errors
