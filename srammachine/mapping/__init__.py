"""Public DeepSeek decode hardware-mapping API."""

from srammachine.inference import MoEParallelStrategy

from .model import (
    DeepSeekV3Config,
    DeepSeekV32Config,
    ModelConfig,
    load_model_config,
)
from .mapper import (
    HardwareMapper,
    HardwareMappingRequest,
    HardwareMappingResult,
    OperatorHardwareMapping,
)

__all__ = [
    "ModelConfig",
    "DeepSeekV3Config",
    "DeepSeekV32Config",
    "load_model_config",
    "MoEParallelStrategy",
    "HardwareMappingRequest",
    "HardwareMappingResult",
    "OperatorHardwareMapping",
    "HardwareMapper",
]
