import libimage
import tifffile
import numpy as np
import tike.ptycho
import tike.ptycho.learn

def test_construct_simulated_training_set(W=1024, N=200, S=64):
    fov = libimage.load('coins', W) * np.exp(1j * libimage.load('earring', W))
    assert np.abs(fov).max() <= 1.0
    scan = np.random.uniform(1, W-S-1, (N, 2))
    probe = 1000 * (tike.ptycho.probe.gaussian(S)[None, None, None, ...]).astype('complex64')

    patches = tike.ptycho.learn.extract_patches(
        psi=fov,
        scan=scan,
        patch_width=probe.shape[-1],
    ).astype('complex64')
    print(patches.dtype, patches.shape)
    np.save('patches', patches)

    diffraction = np.fft.ifftshift(tike.ptycho.simulate(
        detector_shape=probe.shape[-1],
        probe=probe,
        scan=scan,
        psi=fov,
    ), axes=(-2, -1),).astype('float32')
    print(diffraction.dtype, diffraction.shape)
    np.save('diffraction', diffraction)
    tifffile.imwrite('diffraction.tiff', diffraction[N//2])

    print(f'Training params = {np.prod(diffraction.shape)}')

if __name__ == '__main__':
    test_construct_simulated_training_set()
