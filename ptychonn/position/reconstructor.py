import torch
import numpy as np

from ptychonn.position.configs import InferenceConfig
from ptychonn.position.io import *


class VirtualReconstructor:
    def __init__(self, configs: InferenceConfig):
        self.configs = configs
        self.device = None
        self.object_image_array = None

    def build(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

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
