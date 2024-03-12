import os
import logging
import numpy as np

import ptychonn.pospred
from ptychonn.pospred.configs import InferenceConfigDict
from ptychonn.pospred.core import ProbePositionCorrectorChain
from ptychonn.pospred.position_list import ProbePositionList


logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)


def test_multiiter_pos_calculation():
    scan_idx = 235

    configs = InferenceConfigDict(
        reconstruction_image_path=os.path.join('data', 'pospred', 'pred_test{}'.format(scan_idx), 'pred_phase.tiff'),
        random_seed=196,
        debug=False,
        probe_position_list=None,
        # To start with an initial position set, do the folliwing instead:
        # probe_position_list=ProbePositionList(position_list=arr),
        # where arr is a [N, 2] array of probe positions in pixel.
        central_crop=None,
        num_neighbors_collective=4,
    )
    # One can use different values for a config key at different iterations. To do this, create dict keys in the
    # config object with a name of "<existing_config_ket_name>_multiiter". For example:
    # configs.__dict__['method_multiiter'] = ["serial", "collective", "collective"]
    # configs.__dict__['hybrid_registration_tols_multiiter'] = [[0.3, 0.15], [0.15, 0.1], [0.15, 0.1]]
    # Alternatively, you may also put these multiiter keys in a json or toml and read them.
    configs.load_from_toml(os.path.join('data', 'pospred', 'config_{}.toml'.format(scan_idx)))
    print(configs)

    corrector_chain = ProbePositionCorrectorChain(configs)
    corrector_chain.verbose = False
    corrector_chain.build()
    corrector_chain.run()

    calc_pos_list = corrector_chain.corrector_list[-1].new_probe_positions.array

    gold_pos_list = np.genfromtxt(os.path.join('data_gold', 'pospred', 
                                               'calc_pos_235.csv'),
                                  delimiter=',')
    gold_pos_list = gold_pos_list / 8e-9
    calc_pos_list -= np.mean(calc_pos_list, axis=0)
    gold_pos_list -= np.mean(gold_pos_list, axis=0)
    print(gold_pos_list, calc_pos_list)
    assert np.allclose(calc_pos_list, gold_pos_list, atol=1e-1)


if __name__ == '__main__':
    test_multiiter_pos_calculation()

