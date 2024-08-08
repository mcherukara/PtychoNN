"""Ptychography probe position prediction with PtychoNN

A ptychography probe position prediction algorithm making use of PtychoNN [1],
a single-shot phase retrieval network.

## How it works

A trained PtychoNN model is able to predict the local phase around a scan point
given the diffraction pattern alone, not needing any overlapping diffraction
patterns or position information. With predictions made around each scan point,
this algorithm finds out their pairwise offsets using common (but customized)
image registration methods. It then finds a least-squares solution of the
positions of all points in the same coordinates system by solving a linear
equation.

## How to use it with PtychoNN

### Prediction

Run prediction on diffraction patterns in PtychoNN and save all the images as a
single 3D tiff file.

### Create run configurations

Create an `InferenceConfig` object and set `reconstruction_image_path` and
other parameters like `num_neighbors_collective`, `method`, etc. Use the
default parameters as a starting point.

Image registration parameters are supplied in a `RegistrationConfig` object
to `registration_params` of `InferenceConfig`, for example:

```python
configs = InferenceConfig(
    ...
    registration_params=RegistrationConfig(
        registration_method='error_map',
        ...
    )
)
```
If the settings for `RegistrationConfig` are stored in and read from a JSON
or TOML file, just put these parameters at the **same level** as other
parameters. Don't create nested structures in config files.

To start with an initial position set, create a `ProbePositionList` object with
the initial positions, and pass this object to the config obejct:

```python
configs = InferenceConfig(
    ...
    probe_position_list=ProbePositionList(position_list=arr)
)
```
where arr is a `(N, 2)` array of probe positions in pixel.

Using the `ProbePositionCorrectorChain` class allows one to run position
prediction for multiple iterations with varied settings for certain parameters
for each iteration. To do this, create keys in the config object named as
`<name_of_existing_key>_multiiter`, and set a list or tuple of values to it.
Each element of the list is the value for that iteration. For example, setting
`configs.__dict__['method_multiiter'] = ["serial", "collective", "collective"]`
would tell the corrector chain to run 3 iterations, with `method` set to
`"serial"`, `"collective"`, and `"collective"` respectively.

> **Note:** do not pass "_multiiter" keys to the config object's constructor as it will not be recognized.
> Instead, either create new keys in the config object's dictionary container (`configs.__dict__`) after
> the object is instantiated,
> or keep these settings in a JSON or TOML file and read them afterwards.

### Run prediction

Just run the following:

```python
corrector_chain = ProbePositionCorrectorChain(configs)
corrector_chain.build()
corrector_chain.run()
```
Predicted positions can be obtained from
`corrector_chain.corrector_list[-1].new_probe_positions.array`.

### Examples

`tests/test_multiiter_pos_calculation.py` shows an example of a 3-iteration
position prediction run with images already predicted by PtychoNN. The script
demonstrates a case without any initial position input; however, if an initial
position set is desired, one can provide that through the `position_list` key
of the config object. See comments in the config object constructor inside the
script.

**References**

1. M. J. Cherukara, T. Zhou, Y. Nashed, P. Enfedaque, A. Hexemer, R. J. Harder, M. V. Holt, AI-enabled high-resolution scanning coherent diffraction imaging. Appl Phys Lett 117, 044103 (2020).
"""

from .configs import InferenceConfig, RegistrationConfig
from .core import ProbePositionCorrectorChain, PtychoNNProbePositionCorrector
from .position_list import ProbePositionList
