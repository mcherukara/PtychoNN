from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychonn")
except PackageNotFoundError:
    # package is not installed
    pass

from ptychonn._infer.__main__ import infer, stitch_from_inference, Tester
from ptychonn._train.__main__ import Trainer
from ptychonn.model import *
from ptychonn.plot import *
