import pathlib
import importlib.resources

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import torch
import tqdm

import ptychonn._model as helper_small_model


def stitch_from_data(inferences, base_dir, pix=None):
    ## parameters required for stitching individual inferences
    spiral_step = 0.05

    spiral_traj = np.load(base_dir / 'src/optimized_route.npz')  ##
    step = spiral_step * -1e-6
    pos_x, pos_y = spiral_traj['x'] * step, spiral_traj['y'] * step

    x = np.arange(pos_x.min() - 0.5e-6, pos_x.max() + 0.5e-6, 10e-9)
    y = np.arange(pos_y.min() - 0.5e-6, pos_y.max() + 0.5e-6, 10e-9)

    result = np.zeros((y.shape[0], x.shape[0]))
    cnt = np.copy(result)
    cnt1 = cnt + 1

    xx = np.arange(128) * 10e-9
    xx -= xx.mean()
    yy = np.copy(xx)

    result = np.zeros((375, 375))
    tmp_view = np.zeros((963, 375, 375))

    for i in tqdm.tqdm(range(0, 963), leave=False):
        data_ = inferences[i]
        xxx = xx + pos_x[i]
        yyy = yy + pos_y[i]

        if pix is not None:
            xxx = xxx[pix:-pix]
            yyy = yyy[pix:-pix]
            data_ = data_[pix:-pix, pix:-pix]
        find_pha = scipy.interpolate.interp2d(xxx,
                                              yyy,
                                              data_,
                                              kind='linear',
                                              fill_value=0)
        tmp = find_pha(x, y)
        cnt += tmp != 0
        result += tmp
        tmp_view[i] = tmp[:, ::-1]
    #stitch_im_803[nfiles,:,:] = (result/np.maximum(cnt, cnt1))[:,::-1]
    stitched = result / np.maximum(cnt, cnt1)
    return stitched


@click.command()
@click.argument(
    'base_dir',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'out_path',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
def infer(
    base_dir: pathlib.Path,
    out_path: pathlib.Path,
):
    '''Run the inference workflow.'''

    NGPUS = torch.cuda.device_count()
    BATCH_SIZE = NGPUS * 64

    with importlib.resources.path(
            'ptychonn._infer',
            'weights.pth',
    ) as load_model_path:
        print('Model path exist:', load_model_path.exists())
        print('Setting up the inferences...')

        # the test inferences
        print(f'Loading model at {load_model_path}')
        recon_model = helper_small_model.ReconSmallPhaseModel()
        tester = helper_small_model.Tester(
            model=recon_model,
            batch_size=BATCH_SIZE,
            model_params_path=load_model_path,
        )

    # Trying the inference for scan_111_000080
    data_path = base_dir / 'src/scan_506_000793.h5'
    inferences_out_file = base_dir / 'out/inferences_506.npz'
    print('Does data path exist?', data_path.exists())

    with h5py.File(data_path) as f:
        X_test_data = f['entry/data/data'][()]

    tester.setTestData(X_test_data)
    inferences = tester.predictTestData(npz_save_path=inferences_out_file)
    print(f'Finished the inference stage and saved at', inferences_out_file)

    ### Loading stitched data from saved file

    print('Loading inferences from', inferences_out_file)
    inferences = np.load(inferences_out_file)['ph']
    stitched = stitch_from_data(inferences, base_dir)
    plt.figure(1, figsize=[8.5, 7])
    plt.pcolormesh(stitched)
    #plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.tight_layout()
    plt.title('stitched_inferences')
    plt.savefig('stitched_506.png', bbox_inches='tight')
    plt.show()

    test_inferences = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 4, figsize=[13, 3])
    for ix, inf in enumerate(test_inferences):
        plt.subplot(1, 4, ix + 1)
        plt.pcolormesh(inferences[inf])
        plt.colorbar()
        plt.title('Inference at position {0}'.format(inf))
    plt.tight_layout()
    plt.savefig('inferences_0_to_4_scan506.png', bbox_inches='tight')
    plt.show()

    return 0
