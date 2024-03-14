import os.path
import warnings

import torch
import numpy as np

from ptychonn.position.configs import InferenceConfigDict
from ptychonn.position.io import *


class Reconstructor:
    def __init__(self, config_dict: InferenceConfigDict):
        """
        Inference engine for PtychoNN.

        :param config_dict: dict. Configuration dictionary.
        """
        self.config_dict = config_dict
        self.device = None

    def build(self):
        if self.config_dict.cpu_only or (not torch.cuda.is_available()):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

    def batch_infer(self, x):
        pass


class VirtualReconstructor(Reconstructor):
    def __init__(self, config_dict: InferenceConfigDict):
        super().__init__(config_dict)
        self.object_image_array = None

    def set_object_image_array(self, arr):
        self.object_image_array = arr

    def batch_infer(self, x):
        """
        Here x is supposed to be a list of indices for which the object images are to be retrieved.

        :param x: list[int].
        :return: np.ndarray.
        """
        a = np.take(self.object_image_array, indices=x, axis=0)
        return a, a
