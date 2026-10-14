"""V2.1 tactile additions for the colleague WAM-VA baseline."""

from .tactile_codec import FrozenUniversalTactileVAE, UniversalTactileTaskConfig
from .tactile_expert import UniversalTactileExpert, UniversalTactileExpertConfig

__all__ = [
    "FrozenUniversalTactileVAE",
    "UniversalTactileExpert",
    "UniversalTactileExpertConfig",
    "UniversalTactileTaskConfig",
]
