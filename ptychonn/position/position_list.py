import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ptychonn.position.io import load_probe_positions_from_file, save_positions_to_csv


class ProbePositionList:
    def __init__(
        self,
        file_path=None,
        position_list=None,
        unit="pixel",
        psize_nm=None,
        convert_to_pixel=True,
        first_is_x=False,
    ):
        """
        Probe position list.

        :param file_path: str.
        :param position_list: np.ndarray.
        :param unit: str. Original unit of the data.
        :param psize_nm: float. Real-space pixel size in nm.
        """
        if file_path is not None:
            array = load_probe_positions_from_file(file_path, first_is_x=first_is_x)
        else:
            array = position_list
        self.array = array
        self.original_unit = unit
        self.psize_nm = psize_nm
        if convert_to_pixel:
            self.convert_position_unit_to_px()

    def convert_position_unit_to_px(self):
        """
        Convert the unit of position values to pixel.
        """
        if self.original_unit == "pixel":
            return
        factor = {"m": 1e9, "cm": 1e7, "mm": 1e6, "um": 1e3, "nm": 1}[
            self.original_unit
        ]
        self.array = self.array * factor / self.psize_nm

    def __len__(self):
        return len(self.array)

    def shape(self):
        return self.array.shape

    def copy_with_zeros(self):
        a = copy.deepcopy(self)
        a.array = np.zeros_like(self.array)
        return a

    def plot(self, show=True, return_obj=False):
        cmap = matplotlib.cm.get_cmap("Spectral")
        color_list = [
            matplotlib.colors.rgb2hex(cmap(x))
            for x in np.linspace(0, 1, self.array.shape[0])
        ]
        fig, ax = plt.subplots(1, 1)
        scat = plt.scatter(
            self.array[:, 1],
            self.array[:, 0],
            c=color_list,
            s=1,
        )
        lines = plt.plot(
            self.array[:, 1], self.array[:, 0], linewidth=0.5, alpha=0.3, c="gray"
        )
        plt.gca().invert_yaxis()
        if show:
            plt.show()
        if return_obj:
            return fig, ax, scat

    def to_csv(self, filename, unit="m", psize_nm=1):
        arr = self.array
        if unit != "pixel":
            arr = arr * psize_nm
            factor = {"m": 1e9, "cm": 1e7, "mm": 1e6, "um": 1e3, "nm": 1}[unit]
            arr = arr / factor
        save_positions_to_csv(arr, filename)
