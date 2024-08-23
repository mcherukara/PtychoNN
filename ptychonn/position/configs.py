import json
import dataclasses
import warnings
from typing import Any, Optional, Union, Literal
from collections.abc import Sequence

import numpy as np

from ptychonn.position.position_list import ProbePositionList
from ptychonn.position.io import DataFileHandle

try:
    import tomli
except ImportError:
    warnings.warn("Unable to import tomli, which is needed to load a TOML config file.")


@dataclasses.dataclass
class Config:
    def __str__(self, *args, **kwargs):
        s = ""
        for key in self.__dict__.keys():
            s += "{}: {}\n".format(key, self.__dict__[key])
        return s

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
                    v = "_".join([str(x) for x in v])
                else:
                    v = str(v)
            d[key] = v
        return d

    @staticmethod
    def recursive_query(config_obj, key):
        for k in config_obj.__dict__.keys():
            if k == key:
                return config_obj.__dict__[k]
        for k, item in config_obj.__dict__.items():
            if isinstance(item, Config):
                return config_obj.recursive_query(item, key)
        return

    def query(self, key):
        return self.recursive_query(self, key)

    @staticmethod
    def overwrite_value_to_key(config_obj, key, value):
        """
        Recursively search a Config and any of its keys that are also objects of Config for `key`.
        Replace its value with `value` if found.
        """
        is_multiiter_key = key.endswith("_multiiter")
        key_basename = key if not is_multiiter_key else key[: -len("_multiiter")]
        for k in config_obj.__dict__.keys():
            if k == key_basename:
                config_obj.__dict__[key] = value
                return
        for k, item in config_obj.__dict__.items():
            if isinstance(item, Config):
                config_obj.overwrite_value_to_key(item, key, value)
        return

    def dump_to_json(self, filename):
        f = open(filename, "w")
        d = self.get_serializable_dict()
        json.dump(d, f)
        f.close()

    def load_from_json(self, filename):
        """
        This function only overwrites entries contained in the JSON file. Unspecified entries are unaffected.
        """
        f = open(filename, "r")
        d = json.load(f)
        for key in d.keys():
            self.overwrite_value_to_key(self, key, d[key])
        f.close()

    def load_from_toml(self, filename):
        """
        This function only overwrites entries contained in the TOML file. Unspecified entries are unaffected.
        """
        f = open(filename, "rb")
        d = tomli.load(f)
        for key in d.keys():
            self.overwrite_value_to_key(self, key, d[key])
        f.close()

    @staticmethod
    def from_toml(filename):
        obj = InferenceConfig()
        obj.load_from_toml(filename)
        return obj

    @staticmethod
    def from_json(filename):
        obj = InferenceConfig()
        obj.load_from_json(filename)
        return obj


@dataclasses.dataclass
class RegistrationConfig(Config):
    registration_method: Literal["error_map", "sift", "hybrid"] = "error_map"
    """Registration method. Can be "error_map", "sift", "hybrid"."""

    max_shift: int = 7
    """The maximum x/y shift allowed in error map."""

    do_subpixel: bool = True
    """If True, error map algorithm will attempt to get subpixel precision through quadratic fitting."""

    subpixel_fitting_window_size: int = 5
    """Window size for subpixel fitting."""

    subpixel_diff_tolerance: float = 2.0
    """
    If the x or y distance between the subpixel offset found and the integer offset is beyond this value, subpixel
    result will be rejected and integer offset will be used instead.
    """

    subpixel_fitting_check_coefficients: bool = True
    """
    If True, coefficients of the fitted quadratic function are checked and the result will be marked questionable
    if the quadratic function looks too smooth.
    """

    sift_outlier_removal_method: Literal["trial_error", "kmeans", "isoforest", "ransac"] = "trial_error"
    """Method for detecting outlier matches for SIFT. Can be "trial_error", "kmeans", "isoforest", "ransac"."""

    sift_border_exclusion_length: int = 16
    """
    The length of the near-boundary region of the image. When doing SIFT registration, if a matching pair of
    keypoints involve points in this region, it will be discarded. However, if all matches (after outlier removal)
    are near-boundary, they are used as they are. This operation is less aggressive than `central_crop`.
    """

    registration_downsample: int = 1
    """Image downsampling before registration."""

    hybrid_registration_algs: Sequence[str] = (
        "error_map_expandable",
        "sift",
    )
    """Hybrid registration algorithms"""

    hybrid_registration_tols: Sequence[float] = (0.15, 0.3)
    """Hybrid registration tolerances. This value is disregarded unless registration method is hybrid."""

    nonhybrid_registration_tol: float = None
    """Error tolerance for non-hybrid registration. This value is disregarded if registration method is hybrid."""

    registration_tol_schedule: Optional[Sequence[Sequence[int, float], ...]] = None
    """
    The schedule of error tolerance for registration algorithms. This should be a (N, 2) list. In each sub-list,
    the first value is the index of point, and the second value is the new tolerance value to be used at and
    after that point.
    """

    min_roi_stddev: float = 0.2
    """
    The minimum standard deviation required in the region where registration errors are calculated. If the standard
    deviation is below this value, registration result will be rejected as the area for error check might be too
    flat to be conclusive.
    """

    use_baseline_offsets_for_points_on_same_row: bool = False
    """
    If True, baseline offset's x-component will be used for the horizontal offsets of all points on the same
    row if they are arranged in a rectangular grid.
    """

    use_baseline_offsets_for_unregistered_points: bool = False
    """
    If True, if a point is not successfully registered with any neighbor in collective mode, it will fill
    the linear system with the offsets of the two adjacently indexed points to that point from baseline positions.
    """

    use_baseline_offsets_for_uncertain_pairs: bool = False
    """
    If True, if an image pair looks too empty to provide reliable registration result, it will fill
    the linear system with the offsets of the two adjacently indexed points to that point from baseline positions.
    """

    use_fast_errormap: bool = False
    """
    Use fast error map algorithm, where errors are calculated between a sliced region of image 1 and a cropped
    version of image 2, instead of rolling image 2 to impose shift.
    """

    errormap_error_check_tol: float = 0.3
    """Error map result will be marked questionable if the lowest error is beyond this value."""


@dataclasses.dataclass
class InferenceConfig(Config):
    # ===== General configs =====
    registration_params: RegistrationConfig = dataclasses.field(
        default_factory=RegistrationConfig
    )
    """Registration parameters."""

    reconstruction_image_path: str = ''
    """
    Path to the reconstructed images to be used for position prediction. If empty, then
    `reconstruction_images` must be provided.
    """
    
    reconstruction_images: Optional[np.ndarray] = None
    """
    Reconstructed images. Ignored if `reconstruction_image_path` is provided.
    """

    probe_position_list: Optional[ProbePositionList] = None
    """
    A ProbePositionList object used for finding nearest neighbors in collective mode.
    If None, `probe_position_data_path` must be provided.
    """

    probe_position_data_path: Optional[str] = None
    """
    Path to the data file containing probe positions, which should be a CSV file with each line containing the 
    positions in y and x. Ignored if `probe_position_list` is provided. 
    """

    probe_position_data_unit: Optional[str] = None
    """Unit of provided probe position. Can be 'nm', 'm', or 'pixel'. Ignored if `probe_position_list` is provided."""

    pixel_size_nm: Optional[float] = None
    """Pixel size of input positions. Ignored if `probe_position_list` is provided."""

    baseline_position_list: Optional[ProbePositionList] = None
    """Baseline positions. Used by ProbePositionCorrectorChain when the serial mode result is bad."""

    central_crop: Optional[Sequence[int, int]] = None
    """
    List or tuple of int. Patch size used for image registration. If smaller than the reconstructed object size,
    a patch will be cropped from the center.
    """

    method: Literal["serial", "collective"] = "collective"
    """Method for correction. Can be 'serial' or 'collective'"""

    num_neighbors_collective: int = 3
    """Number of neighbors in collective registration"""

    offset_estimator_order: int = 1
    """
    Order of momentum used in the offset estimator. The estimator is used only in serial mode and when the
    registration result is not reliable.
    """

    offset_estimator_beta: float = 0.5
    """Weight of past offsets when updating the running average of offsets in the offset estimator."""

    smooth_constraint_weight: float = 1e-2
    """
    Weight of the smoothness constraint when solving for global-frame probe positions. This is the lambda_2
    in the equation in the paper.
    """

    rectangular_grid: bool = False
    """
    Whether the scan grid is a rectangular grid. Some parameters including
    `use_baseline_offsets_for_points_on_same_row` won't take effect unless this is set to True.
    """

    random_seed: Optional[int] = 123
    """Random seed."""

    debug: bool = False
