import importlib.resources
import pathlib
import typing

from torch.utils.data import TensorDataset, DataLoader
import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import torch
import tqdm

import ptychonn.model


def stitch_from_inference(
    inferences: npt.NDArray,
    scan: npt.NDArray,
    *,
    stitched_pixel_width: float,
    inference_pixel_width: float,
    pix: int = 0,
) -> npt.NDArray:
    '''Combine many overlapping inferred patches into a single image.

    Parameters
    ----------
    inferences : (POSITION, WIDTH, HEIGHT)
        Overlapping patches of the inferred field of view.
    scan : (POSITION, 2) [m]
        The coordinates of each of the overlapping patches.
    pix : int
        Shrink the interpolation region by this number of pixels.
    stitched_pixel_width : float [m]
        The width of a pixel in the stitched image.
    inference_pixel_width : float [m]
        The width of a pixel in the inferred image patches.

    Returns
    -------
    stitched : (COMBINED_WIDTH, COMBINED_HEIGHT) np.array
        The stitched together image.
    '''
    pos_x = scan[..., 1]
    pos_y = scan[..., 0]

    # The global axes of the stitched image in meters
    x = np.arange(pos_x.min(),
                  pos_x.max() + inferences.shape[-1] * inference_pixel_width,
                  step=stitched_pixel_width)
    y = np.arange(pos_y.min(),
                  pos_y.max() + inferences.shape[-1] * inference_pixel_width,
                  step=stitched_pixel_width)

    result = np.zeros((y.shape[0], x.shape[0]))
    cnt = np.zeros_like(result)

    # The local axes of the inference patches meters
    xx = np.arange(inferences.shape[-1]) * inference_pixel_width

    for i in tqdm.tqdm(range(len(inferences)), leave=False):
        data_ = inferences[i]
        xxx = xx + pos_x[i]
        yyy = xx + pos_y[i]

        if pix > 0:
            xxx = xxx[pix:-pix]
            yyy = yyy[pix:-pix]
            data_ = data_[pix:-pix, pix:-pix]
        find_pha = scipy.interpolate.interp2d(
            xxx,
            yyy,
            data_,
            kind='linear',
            fill_value=0,
        )
        tmp = find_pha(x, y)
        cnt += tmp != 0
        result += tmp

    return result / np.maximum(cnt, 1)


@click.command(name='infer')
@click.argument(
    'data_path',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'scan_path',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'out_dir',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
def infer_cli(
    data_path: pathlib.Path,
    scan_path: pathlib.Path,
    out_dir: pathlib.Path,
):
    '''Infer a reconstructed image from diffraction patterns at DATA_PATH and
    scan positions at SCAN_PATH. Save inferred patches and stitched image at
    OUT_DIR.
    '''

    inferences_out_file = out_dir / 'inferences_506.npz'
    click.echo(f'Does data path exist? {data_path.exists()}')

    with h5py.File(data_path) as f:
        data = f['entry/data/data'][()]

    inferences = infer(
        data=data,
        inferences_out_file=inferences_out_file,
    )

    ## parameters required for stitching individual inferences
    spiral_step = 0.05
    step = spiral_step * -1e-6
    spiral_traj = np.load(scan_path)
    scan = np.stack((spiral_traj['x'], spiral_traj['y']), axis=-1) * step
    stitched = stitch_from_inference(
        inferences,
        scan,
        stitched_pixel_width=10e-9,
        inference_pixel_width=10e-9,
    )

    # Plotting some summary images
    plt.figure(1, figsize=[8.5, 7])
    plt.pcolormesh(stitched)
    plt.colorbar()
    plt.tight_layout()
    plt.title('stitched_inferences')
    plt.savefig(out_dir / 'stitched_506.png', bbox_inches='tight')

    test_inferences = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 4, figsize=[13, 3])
    for ix, inf in enumerate(test_inferences):
        plt.subplot(1, 4, ix + 1)
        plt.pcolormesh(inferences[inf])
        plt.colorbar()
        plt.title('Inference at position {0}'.format(inf))
    plt.tight_layout()
    plt.savefig(out_dir / 'inferences_0_to_4_scan506.png', bbox_inches='tight')

    return 0


def infer(
    data: npt.NDArray,
    *,
    inferences_out_file: typing.Optional[pathlib.Path] = None,
) -> npt.NDArray:
    '''Infer ptychography reconstruction for the given data.

    For each diffraction pattern in data, the corresponding patch of object is
    reconstructed.

    Set the CUDA_VISIBLE_DEVICES environment variable to control which GPUs
    will be used.

    Parameters
    ----------
    data : (POSITION, WIDTH, HEIGHT)
        Diffraction patterns to be reconstructed.
    inferences_out_file : pathlib.Path
        Optional file to save reconstructed patches.

    Returns
    -------
    inferences : (POSITION, WIDTH, HEIGHT)
        The reconstructed patches inferred by the model.
    '''
    tester = Tester()
    tester.setTestData(
        data,
        batch_size=max(torch.cuda.device_count(), 1) * 64,
    )
    return tester.predictTestData(npz_save_path=inferences_out_file)


class Tester():
    '''
    '''

    def __init__(
        self,
        *,
        model: typing.Optional[torch.nn.Module] = None,
        model_params_path: typing.Optional[pathlib.Path] = None,
    ):
        self.model = ptychonn.model.ReconSmallModel(
        ) if model is None else model

        if model_params_path is None:
            with importlib.resources.path(
                    'ptychonn._infer',
                    'weights.pth',
            ) as model_params_path:
                self.model.load_state_dict(torch.load(model_params_path))
        else:
            self.model.load_state_dict(torch.load(model_params_path))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

    def setTestData(self, X_test: np.ndarray, batch_size: int):
        self.X_test = torch.tensor(X_test[:, None, ...].astype('float32'))
        self.test_data = TensorDataset(self.X_test)
        self.testloader = DataLoader(self.test_data,
                                     batch_size=batch_size,
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
