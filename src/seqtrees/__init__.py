"""Sequential tree models for tabular synthetic data generation."""

from importlib.metadata import PackageNotFoundError, version

from .synthesizer import SequentialTreeSynthesizer

try:
    __version__ = version("seqtrees")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["SequentialTreeSynthesizer", "__version__"]
