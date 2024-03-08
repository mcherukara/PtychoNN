import logging
logging.getLogger(__name__).setLevel(logging.INFO)

import collections
import json
import dataclasses
from typing import Any


@dataclasses.dataclass

class ConfigDict(collections.defaultdict):
    def __str__(self, *args, **kwargs):
        s = ''
        for key in self.__dict__.keys():
            s += '{}: {}\n'.format(key, self.__dict__[key])
        return s

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    def get_serializable_dict(self):
        d = {}
        for key in self.__dict__.keys():
            v = self.__dict__[key]
            if not self.__class__.is_jsonable(v):
                if isinstance(v, (tuple, list)):
                    v = '_'.join([str(x) for x in v])
                else:
                    v = str(v)
            d[key] = v
        return d

    def dump_to_json(self, filename):
        try:
            f = open(filename, 'w')
            d = self.get_serializable_dict()
            json.dump(d, f)
            f.close()
        except:
            print('Failed to dump json.')

    def load_from_json(self, filename):
        """
        This function only overwrites entries contained in the JSON file. Unspecified entries are unaffected.
        """
        f = open(filename, 'r')
        d = json.load(f)
        for key in d.keys():
            self.__dict__[key] = d[key]
        f.close()


@dataclasses.dataclass

class InferenceConfigDict(ConfigDict):

    # ===== PtychoNN configs =====
    batch_size: Any = 1
    """Inference batch size."""

    model_path: Any = None
    """Path to a trained PtychoNN model."""

    model: Any = None
    """


    The model. Should be a tuple(nn.Module, kwargs): the first element of the tuple is the class handle of a
    model class, and the second is a dictionary of keyword arguments. The model will be instantiated using these.
    This value is used to instantiate a model object, whose weights are overwritten with those read from


    `model_path`. The provided model class and arguments must match the model being loaded.
    """

    ptycho_reconstructor: Any = None
    """
    Should be either None or a Reconstructor object. If None, PyTorchReconstructor is used by default.
    """

    dp_data_path: Any = None
    """
    The path to the diffraction data file. When using a VirtualReconstrutor that uses already-reconstructed images, 
    keep this as None.
    """

    prediction_output_path: Any = None
    """Path to save PtychoNN prediction results."""

    cpu_only: Any = False

    onnx_mdl: Any = None
    """ONNX file when using ONNXReconstructor."""

    # ===== Image registration configs =====
    registration_method: Any = 'error_map'

    do_subpixel: Any = True

    use_fast_errormap: Any = False

    sift_outlier_removal_method: Any = 'kmeans'
    """Method for detecting outlier matches for SIFT. Can be "trial_error", "kmeans", "isoforest", "ransac"."""

    sift_border_exclusion_length: Any = 16
    """
    The length of the near-boundary region of the image. When doing SIFT registration, if a matching pair of
    keypoints involve points in this region, it will be discarded. However, if all matches (after outlier removal)
    are near-boundary, they are used as they are. This operation is less aggressive than `central_crop`.
    """

    registration_downsample: Any = 1
    """Image downsampling before registration."""

    hybrid_registration_algs: Any = ('error_map_multilevel', 'error_map_expandable', 'sift')
    """Hybrid registration algorithms"""

    hybrid_registration_tols: Any = (0.15, 0.3, 0.3)
    """Hybrid registration tolerances. This value is disregarded unless registration method is hybrid."""

    nonhybrid_registration_tol: Any = None
    """Error tolerance for non-hybrid registration. This value is disregarded if registration method is hybrid."""

    registration_tol_schedule: Any = None
    """
    The schedule of error tolerance for registration algorithms. This should be a (N, 2) list. In each sub-list,
    the first value is the index of point, and the second value is the new tolerance value to be used at and
    after that point.
    """

    min_roi_stddev: Any = 0.2

    subpixel_fitting_window_size: Any = 5

    subpixel_diff_tolerance: Any = 2

    subpixel_fitting_check_coefficients: Any = True

    errormap_error_check_tol: Any = 0.3

    # ===== General configs =====
    reconstruction_image_path: Any = None
    """
    Path to the reconstructed images to be used for position prediction. If None, PtychoNNProbePositionCorrector
    would then require a Reconstructor object that generate reconstructed images from diffraction patterns. 
    Alternatively, one could also keep this argument as None and pass a VirtualReconstructor set to read 
    the reconstructed images to ptycho_reconstructor.
    """

    dp_data_file_path: Any = None

    dp_data_file_handle: Any = None
    """Used as an alternative to `dp_data_file_path`. Should be a `DataFileHandle` object."""

    probe_position_list: Any = None
    """
    A ProbePositionList object used for finding nearest neighbors in collective mode.
    If None, `probe_position_data_path` must be provided.
    """

    probe_position_data_path: Any = None

    probe_position_data_unit: Any = None

    pixel_size_nm: Any = None

    baseline_position_list: Any = None
    """Baseline positions. Used by ProbePositionCorrectorChain when the serial mode result is bad."""

    central_crop: Any = None
    """
    Patch size used for image registration. If smaller than the reconstructed object size, a patch will
    be cropped from the center.
    """

    method: Any = 'collective'
    """Method for correction. Can be 'serial' or 'collective'"""

    max_shift: Any = 7

    num_neighbors_collective: Any = 3
    """Number of neighbors in collective registration"""

    offset_estimator_order: Any = 1

    offset_estimator_beta: Any = 0.5

    smooth_constraint_weight: Any = 1e-2

    use_baseline_offsets_for_uncertain_pairs: Any = False

    rectangular_grid: Any = False

    use_baseline_offsets_for_points_on_same_row: Any = False

    use_baseline_offsets_for_unregistered_points: Any = False
    """
    If True, if a point is not successfully registered with any neighbor in collective mode, it will fill
    the linear system with the offsets of the two adjacently indexed points to that point from baseline positions.
    """

    stitching_downsampling: Any = 1

    random_seed: Any = 123

    debug: Any = None


class TrainingConfigDict(ConfigDict):
    batch_size_per_process: Any = 64

    num_epochs: Any = 60

    learning_rate_per_process: Any = 1e-3

    optimizer: Any = 'adam'

    """String of optimizer name or the handle of a subclass of torch.optim.Optimizer"""

    model_save_dir: Any = '.'
    """Directory to save trained models"""

    model: Any = None
    """
    The model. The three options are:
    (1) None: the model will be instantiated with the default model class.
    (2) A object of nn.Module: the model object will be used as provided.


    (3) tuple(nn.Module, kwargs): the first element of the tuple is the class handle of a model class, and the
        second is a dictionary of keyword arguments. The model will be instantiated using these.
    """

    l1_weight: Any = 0

    tv_weight: Any = 0


class PtychoNNTrainingConfigDict(TrainingConfigDict):
    height: Any = 256

    width: Any = 256

    num_lines_for_training: Any = 100
    """Number of lines used for training"""

    num_lines_for_testing: Any = 60
    """Number of lines used for testing"""

    num_lines_for_validation: Any = 805
    """Number of lines used for testing"""

    dataset: Any = None
    """A torch.Dataset object"""

    validation_ratio: Any = 0.003
    """Ratio of validation set out of the entire dataset"""

    loss_function: Any = None
    """Can be None (default to L1Loss) or a Callable."""

    dataset_decimation_ratio: Any = 1

    schedule_learning_rate: Any = True

    pretrained_model_path: Any = None

