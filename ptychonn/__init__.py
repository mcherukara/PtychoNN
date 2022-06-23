from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychonn")
except PackageNotFoundError:
    # package is not installed
    pass
