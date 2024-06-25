import libimage
import numpy as np
import matplotlib.pyplot as plt
import tike.ptycho
import tike.ptycho.learn


def test_construct_simulated_training_set(W=2048, N=2048 + 1024, S=128):
    phase = libimage.load('coins', W) - 0.5
    amplitude = 1 - libimage.load('earring', W)
    fov = (amplitude * np.exp(1j * phase * np.pi)).astype('complex64')
    plt.imsave('phase.png', np.angle(fov))
    plt.imsave('amplitude.png', np.abs(fov))
    assert np.abs(fov).max() <= 1.0 + 1e-4, np.abs(fov).max()
    assert np.abs(fov).min() >= 0.0, np.abs(fov).min()

    scan = np.random.uniform(1, W - S - 1, (N, 2))

    probe = (tike.ptycho.probe.gaussian(S)[None, None, None,
                                           ...]).astype('complex64')

    patches = tike.ptycho.learn.extract_patches(
        psi=fov,
        scan=scan,
        patch_width=probe.shape[-1],
    ).astype('complex64')
    print(patches.dtype, patches.shape)

    diffraction = np.fft.ifftshift(
        tike.ptycho.simulate(
            detector_shape=probe.shape[-1],
            probe=probe,
            scan=scan,
            psi=fov[None, ...],
        ),
        axes=(-2, -1),
    ).astype('float32')
    print(diffraction.dtype, diffraction.shape)
    plt.imsave('diffraction.png', np.log10(diffraction[N // 2]))

    print(f'Training params = {np.prod(diffraction.shape)}')

    np.savez_compressed(
        'simulated_data.npz',
        reciprocal=diffraction[1024:],
        real=patches[1024:],
        scan=scan[1024:],
    )
    np.savez_compressed(
        'simulated_test.npz',
        reciprocal=diffraction[:1024],
        real=patches[:1024],
        scan=scan[:1024],
    )

if __name__ == '__main__':
    test_construct_simulated_training_set()
