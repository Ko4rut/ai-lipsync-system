"""Data pipeline package for the GRID Audio-Visual Speech Corpus.

Public surface kept minimal; import from submodules directly for internal use.
"""

from .config import PipelineConfig, USABLE_SPEAKERS, EXCLUDED_SPEAKERS
from .exceptions import GRIDPipelineError
from .state import StateManager
from .split import compute_speaker_split

__all__ = [
    "PipelineConfig",
    "USABLE_SPEAKERS",
    "EXCLUDED_SPEAKERS",
    "GRIDPipelineError",
    "StateManager",
    "compute_speaker_split",
]
