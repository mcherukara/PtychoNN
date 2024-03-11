import logging
logging.getLogger(__name__).setLevel(logging.INFO)

import os

import numpy as np

import ptychonn.pospred
from ptychonn.pospred.configs import InferenceConfigDict
from ptychonn.pospred.core import ProbePositionCorrectorChain


def test_multiiter_pos_calculation():
    scan_idx = 235

    config_dict = InferenceConfigDict(
        reconstruction_image_path=os.path.join('data', 'pospred', 'pred_test{}'.format(scan_idx), 'pred_phase.tiff'),
        random_seed=196,
        debug=False,
        probe_position_list=None,
        central_crop=None
    )
    config_dict.load_from_toml(os.path.join('data', 'pospred', 'config_{}.toml'.format(scan_idx)))
    print(config_dict)

    corrector_chain = ProbePositionCorrectorChain(config_dict)
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

