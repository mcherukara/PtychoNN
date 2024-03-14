import logging
import collections

import numpy as np
import tifffile
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import sklearn.neighbors

from ptychonn.position.configs import ConfigDict, InferenceConfigDict
from ptychonn.position.reconstructor import VirtualReconstructor
from ptychonn.position.io import create_data_file_handle, VirtualDataFileHandle
from ptychonn.position.position_list import ProbePositionList
from ptychonn.position.registrator import Registrator


class OffsetEstimator:
    def __init__(self, beta=0.5, order=2):
        self.beta = beta
        self.beta_step_length = 0.9
        self.order = order
        self.v_bar = np.array([0.0, 0.0])  # Accumulated velocity
        self.a_bar = np.array([0.0, 0.0])  # Accumulated acceleration
        self.delta_a_bar = np.array([0.0, 0.0])  # Accumulated change of acceleration
        self.last_v = np.array([0.0, 0.0])
        self.last_a = np.array([0.0, 0.0])
        self.i_iter = 0
        self.step_length_bar = 0.0

    def update_offset(self, v):
        if self.order == 3:
            if self.i_iter > 2:
                this_a = v - self.last_v
                this_delta_a = this_a - self.last_a
                self.delta_a_bar = (
                    self.beta * self.delta_a_bar + (1 - self.beta) * this_delta_a
                )
                self.last_a = this_a
            elif self.i_iter == 2:
                this_a = v - self.last_v
                self.delta_a_bar = this_a - self.last_a
                self.last_a = this_a
            elif self.i_iter == 1:
                this_a = v - self.last_v
                self.last_a = this_a
            else:
                self.delta_a_bar = np.array([0.0, 0.0])
        if self.order == 2:
            if self.i_iter > 1:
                this_a = v - self.last_v
                self.a_bar = self.beta * self.a_bar + (1 - self.beta) * this_a
            elif self.i_iter == 1:
                self.a_bar = v - self.last_v
            else:
                self.a_bar = np.array([0.0, 0.0])
        if self.order == 1:
            if self.i_iter > 0:
                self.v_bar = self.beta * self.v_bar + (1 - self.beta) * v
            else:
                self.v_bar = v
        self.last_v = v
        self.update_step_length_bar(v)
        self.i_iter += 1

    def update_offset_with_estimated_value(self, v):
        """
        If the offset of the current iteration is estimated, then only update last-iteration values and the
        iteration number, but do not change any accumulated values.

        :param v: np.ndarray.
        """
        if self.order == 3:
            self.last_a = v - self.last_v
        self.last_v = v
        self.i_iter += 1

    def update_step_length_bar(self, v):
        if self.i_iter == 0:
            self.step_length_bar = np.linalg.norm(v)
        else:
            self.step_length_bar = self.beta_step_length * self.step_length_bar + (
                1 - self.beta_step_length
            ) * np.linalg.norm(v)

    def estimate(self):
        this_v = 0.0
        if self.order == 0:
            this_v = self.last_v
        elif self.order == 1:
            this_v = self.v_bar
        elif self.order == 2:
            this_v = self.last_v + self.a_bar
        elif self.order == 3:
            this_a = self.last_a + self.delta_a_bar
            this_v = self.last_v + this_a
        this_v = this_v / np.linalg.norm(this_v) * self.step_length_bar
        return this_v


class PtychoNNProbePositionCorrector:
    def __init__(self, config_dict: InferenceConfigDict):
        self.config_dict = config_dict
        if self.config_dict.reconstruction_image_path is None:
            if self.config_dict.ptycho_reconstructor is None:
                self.ptycho_reconstructor = PyTorchReconstructor(self.config_dict)
            else:
                self.ptycho_reconstructor = self.config_dict.ptycho_reconstructor
        else:
            self.ptycho_reconstructor = VirtualReconstructor(InferenceConfigDict())

        self.dp_data_fhdl = None
        self.orig_probe_positions = None
        self.new_probe_positions = None
        self.n_dps = 0
        self.registrator = None
        self.lmbda_collective = 1e-6
        self.a_mat = np.array([])
        self.b_vec = np.array([])
        self.count_bad_offset = 0
        self.row_index_list = []
        self.unregistered_indices = []

    def build(self):
        if self.config_dict.random_seed is not None:
            logging.debug(
                "Random seed is set to {}.".format(self.config_dict.random_seed)
            )
            np.random.seed(self.config_dict.random_seed)

        if self.config_dict.reconstruction_image_path is not None and isinstance(
            self.ptycho_reconstructor, VirtualReconstructor
        ):
            recons = tifffile.imread(self.config_dict.reconstruction_image_path)
            self.ptycho_reconstructor.set_object_image_array(recons)
        self.ptycho_reconstructor.build()

        if not self.config_dict.dp_data_file_handle:
            if isinstance(self.ptycho_reconstructor, VirtualReconstructor):
                self.dp_data_fhdl = VirtualDataFileHandle(
                    "",
                    dp_shape=self.ptycho_reconstructor.object_image_array.shape[1:],
                    num_dps=self.ptycho_reconstructor.object_image_array.shape[0],
                )
            else:
                self.dp_data_fhdl = create_data_file_handle(
                    self.config_dict.dp_data_path
                )
        else:
            self.dp_data_fhdl = self.config_dict.dp_data_file_handle
        self.n_dps = self.dp_data_fhdl.num_dps

        self.orig_probe_positions = ProbePositionList(
            position_list=np.zeros([self.n_dps, 2])
        )
        if not self.config_dict.probe_position_list:
            if self.config_dict.probe_position_data_path:
                self.orig_probe_positions = ProbePositionList(
                    file_path=self.config_dict.probe_position_data_path,
                    unit=self.config_dict.probe_position_data_unit,
                    psize_nm=self.config_dict.pixel_size_nm,
                )
        else:
            self.orig_probe_positions = self.config_dict.probe_position_list
        self.new_probe_positions = self.orig_probe_positions.copy_with_zeros()

        self.registrator = Registrator(
            self.config_dict.registration_params,
            random_seed=self.config_dict.random_seed,
        )

        if (
            self.config_dict.rectangular_grid
            and self.config_dict.registration_params.use_baseline_offsets_for_points_on_same_row
        ):
            self.build_row_index_list()

    def build_row_index_list(self):
        assert (
            self.orig_probe_positions
        ), "build_row_index_list requires an input position list, but nothing is given."
        pos = self.orig_probe_positions.array
        jump_inds = np.where(np.abs(pos[1:, 0] - pos[:-1, 0]) > 1e-5)[0] - 1
        jump_inds = np.concatenate([jump_inds, [pos.shape[0]]])
        row_inds = []
        for row, jump_ind in enumerate(jump_inds):
            if row == 0:
                row_inds = row_inds + [row] * jump_ind
            else:
                row_inds = row_inds + [row] * (jump_ind - jump_inds[row - 1])
        assert len(row_inds) == pos.shape[0]
        self.row_index_list = np.array(row_inds)

    def run(self):
        if self.config_dict.method == "serial":
            self.run_probe_position_correction_serial()
        elif self.config_dict.method == "collective":
            self.run_probe_position_correction_collective()
        else:
            raise ValueError(
                "Correction method {} is not supported. ".format(
                    self.config_dict.method
                )
            )

    def reconstruct_dp(self, ind=None, dp=None):
        if isinstance(self.ptycho_reconstructor, VirtualReconstructor):
            assert (
                ind is not None
            ), "Since the reconstructor is a VirtualReconstructor, a index must be provided. "
            obj_amp, obj_ph = self.ptycho_reconstructor.batch_infer([ind])
            dp = np.zeros(self.dp_data_fhdl.dp_shape)
        else:
            if dp is None:
                dp = self.dp_data_fhdl.get_dp_by_consecutive_index(ind)
            obj_amp, obj_ph = self.ptycho_reconstructor.batch_infer(
                dp[np.newaxis, :, :]
            )
        # if self.config_dict.debug:
        #     fig, ax = plt.subplots(1, 3)
        #     ax[0].imshow(dp)
        #     ax[0].set_title('DP')
        #     ax[1].imshow(obj_ph[0])
        #     ax[1].set_title('Recon phase')
        #     ax[2].imshow(obj_amp[0])
        #     ax[2].set_title('Recon mag')
        #     plt.show()
        obj_amp, obj_ph = self.crop_center([obj_amp, obj_ph])
        return obj_amp, obj_ph

    def crop_center(self, image_list):
        if self.config_dict.central_crop is None:
            return image_list
        crop_shape = self.config_dict.central_crop
        for i in range(len(image_list)):
            orig_shape = image_list[i].shape[1:]
            start_point = [(orig_shape[j] - crop_shape[j]) // 2 for j in range(2)]
            new_image = image_list[i][
                :,
                start_point[0] : start_point[0] + crop_shape[0],
                start_point[1] : start_point[1] + crop_shape[1],
            ]
            image_list[i] = new_image
        return image_list

    def run_probe_position_correction_serial(self):
        """
        Run serial mode probe position correction.
        """
        offset_tracker = OffsetEstimator(
            beta=self.config_dict.offset_estimator_beta,
            order=self.config_dict.offset_estimator_order,
        )
        previous_obj = self.reconstruct_dp(0)[1][0]
        for ind in trange(1, self.n_dps):
            current_obj = self.reconstruct_dp(ind)[1][0]
            offset = self.registrator.run(previous_obj, current_obj)
            if (
                self.config_dict.registration_params.use_baseline_offsets_for_uncertain_pairs
                and self.registrator.get_status()
                == self.registrator.get_status_code("empty")
            ):
                offset = (
                    self.orig_probe_positions.array[ind]
                    - self.orig_probe_positions.array[ind - 1]
                )
            elif self.registrator.get_status() == self.registrator.get_status_code(
                "bad"
            ):
                offset = offset_tracker.estimate()
                self.count_bad_offset += 1
            if self.config_dict.debug:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(previous_obj)
                ax[1].imshow(current_obj)
                plt.suptitle("Index {} - {}".format(ind, ind - 1))
                plt.show()
                plt.tight_layout()
                print("Offset: {}".format(offset))
            self._update_probe_position_list(ind, offset)
            previous_obj = current_obj
            offset_tracker.update_offset(offset)

    def run_probe_position_correction_collective(self):
        """
        Run collective probe position correction.
        """
        self.build_linear_system_for_collective_correction()
        self.solve_linear_system(
            mode="residue",
            smooth_constraint_weight=self.config_dict.smooth_constraint_weight,
        )
        self.postprocess()

    def get_neightbor_inds(self, i_dp, knn_inds):
        this_neighbors_inds = []
        # Must have the 2 adjacently indexed DPs
        if i_dp == 0:
            this_neighbors_inds = [1]
        elif i_dp == len(self.orig_probe_positions.array) - 1:
            this_neighbors_inds = [i_dp - 1]
        else:
            this_neighbors_inds = [i_dp - 1, i_dp + 1]
        i_knn = 0
        while len(this_neighbors_inds) < self.config_dict.num_neighbors_collective:
            if knn_inds[i_knn] not in this_neighbors_inds:
                this_neighbors_inds.append(knn_inds[i_knn])
            i_knn += 1
        return this_neighbors_inds

    def build_linear_system_for_collective_correction(self):
        self.a_mat = []
        self.b_vec = []
        # A hash table used to keep track of already registered DP pairs. One can then check whether a pair has been
        # registered with O(1) time.
        registered_pair_hash_table = collections.defaultdict(lambda: None)
        nn_engine = sklearn.neighbors.NearestNeighbors(
            n_neighbors=self.config_dict.num_neighbors_collective + 1
        )
        nn_engine.fit(self.orig_probe_positions.array)
        nn_dists, nn_inds = nn_engine.kneighbors(self.orig_probe_positions.array)
        for i_dp, this_orig_pos in enumerate(tqdm(self.orig_probe_positions.array)):
            if self.config_dict.registration_params.registration_tol_schedule:
                self.registrator.update_tol_by_tol_schedule(
                    i_dp, self.config_dict.registration_params.registration_tol_schedule
                )
            this_knn_inds = nn_inds[i_dp, 1:]
            this_neighbors_inds = self.get_neightbor_inds(i_dp, this_knn_inds)

            # `ind` is also provided in case the reconstructor is a `VirtualReconstructor`.
            current_obj = self.reconstruct_dp(
                dp=self.dp_data_fhdl.get_dp_by_raw_index(i_dp), ind=i_dp
            )[1][0]
            for ind_neighbor in this_neighbors_inds:
                # Check whether the current pair of DPs have already been registered.
                if (
                    registered_pair_hash_table["{}_{}".format(ind_neighbor, i_dp)]
                    is not None
                    or registered_pair_hash_table["{}_{}".format(i_dp, ind_neighbor)]
                ):
                    continue
                # Otherwise, run registration and record the values.
                registered_pair_hash_table["{}_{}".format(ind_neighbor, i_dp)] = 1
                neighbor_obj = self.reconstruct_dp(
                    dp=self.dp_data_fhdl.get_dp_by_raw_index(ind_neighbor),
                    ind=ind_neighbor,
                )[1][0]

                if (
                    self.config_dict.registration_params.use_baseline_offsets_for_points_on_same_row
                    and self.row_index_list is not None
                    and self.row_index_list[i_dp] == self.row_index_list[ind_neighbor]
                ):
                    logging.debug(
                        "DP {} and {} are on the same row, so using baseline offset for this pair.".format(
                            i_dp, ind_neighbor
                        )
                    )
                    offset = (
                        self.orig_probe_positions.array[i_dp]
                        - self.orig_probe_positions.array[ind_neighbor]
                    )
                    self.registrator.algorithm.status = (
                        self.registrator.get_status_code("ok")
                    )
                else:
                    offset = self.registrator.run(neighbor_obj, current_obj)
                    # print('{} - {}: {}'.format(i_dp, ind_neighbor, offset))
                    if (
                        self.config_dict.debug
                        and self.registrator.get_status()
                        != self.registrator.get_status_code("empty")
                    ):
                        fig, ax = plt.subplots(1, 2)
                        im = ax[0].imshow(neighbor_obj, vmin=-0.2, vmax=0.2)
                        plt.colorbar(im)
                        im = ax[1].imshow(current_obj, vmin=-0.2, vmax=0.2)
                        plt.colorbar(im)
                        plt.suptitle("Index {} - {}".format(i_dp, ind_neighbor))
                        plt.show()
                        plt.tight_layout()
                        print("Offset: {}".format(offset))
                if (
                    self.config_dict.registration_params.use_baseline_offsets_for_uncertain_pairs
                    and self.registrator.get_status()
                    == self.registrator.get_status_code("empty")
                ):
                    offset = (
                        self.orig_probe_positions.array[i_dp]
                        - self.orig_probe_positions.array[ind_neighbor]
                    )
                # We want to be more strict with collective mode. If a result is less confident, just skip it.
                if self.registrator.get_status() != self.registrator.get_status_code(
                    "ok"
                ):
                    self.count_bad_offset += 1
                    continue
                else:
                    self.b_vec.append(np.array(offset))
                    self.a_mat.append(self._generate_amat_row(i_dp, ind_neighbor))
        self.a_mat = np.stack(self.a_mat)
        self.b_vec = np.stack(self.b_vec)
        if self.config_dict.registration_params.use_baseline_offsets_for_unregistered_points:
            self.fill_gaps_in_linear_system()

    def fill_gaps_in_linear_system(self):
        self.unregistered_indices = np.where(np.count_nonzero(self.a_mat, axis=0) == 0)[
            0
        ]
        if len(self.unregistered_indices) > 0:
            a_appendix = []
            b_appendix = []
            for ind in self.unregistered_indices:
                if ind > 0:
                    a_row = np.zeros(self.a_mat.shape[1])
                    a_row[ind] = 1
                    a_row[ind - 1] = -1
                    b_row = (
                        self.orig_probe_positions.array[ind]
                        - self.orig_probe_positions.array[ind - 1]
                    )
                    a_appendix.append(a_row)
                    b_appendix.append(b_row)
                if ind < self.a_mat.shape[1] - 1:
                    a_row = np.zeros(self.a_mat.shape[1])
                    a_row[ind] = 1
                    a_row[ind + 1] = -1
                    b_row = (
                        self.orig_probe_positions.array[ind]
                        - self.orig_probe_positions.array[ind + 1]
                    )
                    a_appendix.append(a_row)
                    b_appendix.append(b_row)
            a_appendix = np.stack(a_appendix, axis=0)
            b_appendix = np.stack(b_appendix, axis=0)
            self.a_mat = np.concatenate([self.a_mat, a_appendix], axis=0)
            self.b_vec = np.concatenate([self.b_vec, b_appendix], axis=0)

    def solve_linear_system(self, mode="residue", smooth_constraint_weight=1e-3):
        a_mat = self.a_mat
        b_vec = self.b_vec
        if mode == "residue":
            x0_vec = self.orig_probe_positions.array
            b_vec_damped = a_mat.T @ b_vec - a_mat.T @ (a_mat @ x0_vec)
            a_damped = a_mat.T @ a_mat + self.lmbda_collective * np.eye(a_mat.shape[1])
            if smooth_constraint_weight > 0:
                s_mat = self.get_square_roll_matrix()
                a_damped = a_damped + smooth_constraint_weight * s_mat
                b_vec_damped = b_vec_damped - smooth_constraint_weight * s_mat @ x0_vec
            delta_x_vec = np.linalg.inv(a_damped) @ b_vec_damped
            self.new_probe_positions.array = (
                self.orig_probe_positions.array + delta_x_vec
            )
        else:
            self.new_probe_positions.array = np.linalg.pinv(a_mat) @ b_vec

    def postprocess(self):
        if self.config_dict.registration_params.use_baseline_offsets_for_unregistered_points:
            for ind in self.unregistered_indices:
                if ind > 0:
                    baseline_offset = (
                        self.orig_probe_positions.array[ind]
                        - self.orig_probe_positions.array[ind - 1]
                    )
                    self.new_probe_positions.array[ind] = (
                        self.new_probe_positions.array[ind - 1] + baseline_offset
                    )
                else:
                    baseline_offset = (
                        self.orig_probe_positions.array[ind + 1]
                        - self.orig_probe_positions.array[ind]
                    )
                    self.new_probe_positions.array[ind] = (
                        self.new_probe_positions.array[ind + 1] - baseline_offset
                    )

    def get_square_roll_matrix(self):
        s_mat = np.eye(self.a_mat.shape[1]) * 2
        s_mat[0, 0] = 1
        s_mat[-1, -1] = 1
        off_diag_y_inds = tuple(range(s_mat.shape[0] - 1))
        off_diag_x_inds = tuple(range(1, s_mat.shape[0]))
        s_mat[off_diag_y_inds, off_diag_x_inds] = -1
        off_diag_y_inds = tuple(range(1, s_mat.shape[0]))
        off_diag_x_inds = tuple(range(s_mat.shape[0] - 1))
        s_mat[off_diag_y_inds, off_diag_x_inds] = -1
        return s_mat

    def _generate_amat_row(self, this_ind, neightbor_ind):
        a = np.zeros(self.n_dps)
        a[this_ind] = 1
        a[neightbor_ind] = -1
        return a

    def _update_probe_position_list(self, ind, offset):
        raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(
            ind, ravel=True
        )
        prev_raw_ind = self.dp_data_fhdl.get_actual_indices_for_consecutive_index(
            ind - 1, ravel=True
        )
        self.new_probe_positions.array[raw_ind] = (
            self.new_probe_positions.array[prev_raw_ind] + offset
        )


class ProbePositionCorrectorChain:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.corrector_list = []
        self.multiiter_keys = []
        self.n_iters = 1
        self.baseline_pos_list = config_dict.baseline_position_list
        self.collective_mode_offset_tol = 150
        self.verbose = True
        self.redone_with_baseline = False
        self.has_multiiter_key = False

    def build(self):
        self.build_multiiter_entries()

    def find_multiiter_keys(self, config_obj):
        for key in config_obj.__dict__.keys():
            if "multiiter" in key:
                self.n_iters = len(config_obj.__dict__[key])
                self.multiiter_keys.append(key)
                self.has_multiiter_key = True
        for key, value in config_obj.__dict__.items():
            if isinstance(value, ConfigDict):
                self.find_multiiter_keys(value)

    def build_multiiter_entries(self):
        self.has_multiiter_key = False
        self.find_multiiter_keys(self.config_dict)
        if not self.has_multiiter_key:
            raise ValueError(
                "With ProbePositionCorrectorChain, there should be at least one entry in the config dict "
                'that ends with "_multiiter" and is a list whose length equals to the desired number of '
                "iterations. "
            )

    def run(self):
        for iter in range(self.n_iters):
            self.run_correction_iteration(iter)

    def run_correction_iteration(self, iter):
        logging.debug("Now running iteration {}.".format(iter))
        self.update_config_dict(iter)
        if self.verbose:
            print(self.config_dict)
        corrector = PtychoNNProbePositionCorrector(config_dict=self.config_dict)
        corrector.build()
        if self.verbose:
            corrector.orig_probe_positions.plot()
        corrector.run()
        if self.config_dict.method == "collective" and (
            not self.is_collective_result_good(corrector)
        ):
            # Redo iteration using baseline as initialization if result is bad
            logging.debug(
                "The current iteration is using collective mode and the result is unreliable. Attempting to "
                "redo this iteration with baseline positions as initialization..."
            )
            if self.baseline_pos_list:
                self.config_dict.probe_position_list = self.baseline_pos_list
                corrector = PtychoNNProbePositionCorrector(config_dict=self.config_dict)
                corrector.build()
                corrector.run()
                self.redone_with_baseline = True
            else:
                logging.debug("Baseline position is unavailable.")
        self.corrector_list.append(corrector)
        if self.verbose:
            corrector.new_probe_positions.plot()

    def is_collective_result_good(self, corrector):
        calc_pos = corrector.new_probe_positions.array
        offset_len = np.sqrt(np.sum((calc_pos[1:] - calc_pos[:-1]) ** 2, axis=1))
        if np.count_nonzero(offset_len > self.collective_mode_offset_tol):
            return False
        else:
            return True

    def get_ordinary_key_name(self, mikey):
        ind = mikey.find("_multiiter")
        return mikey[:ind]

    def update_config_dict(self, iter, initialize_with_baseline=False):
        for mikey in self.multiiter_keys:
            key = self.get_ordinary_key_name(mikey)
            self.config_dict.overwrite_value_to_key(
                self.config_dict, key, self.config_dict.query(mikey)[iter]
            )
        if iter > 0:
            last_corrector = self.corrector_list[iter - 1]
            last_probe_pos_array = last_corrector.new_probe_positions.array
            probe_pos_list = ProbePositionList(position_list=last_probe_pos_array)
            if initialize_with_baseline:
                if self.baseline_pos_list:
                    probe_pos_list = self.baseline_pos_list
                else:
                    raise ValueError(
                        "Cannot initialize with baseline positions: baseline position list is None."
                    )
            self.config_dict.probe_position_list = probe_pos_list
            logging.debug(
                "Using result from the last iteration to initialize probe position array..."
            )
