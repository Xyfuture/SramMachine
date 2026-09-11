"""Public MLA/MoE decode hardware-mapping API."""

from srammachine.inference import MoEParallelStrategy

from .model import (
    DeepSeekV3Config,
    DeepSeekV32Config,
    GLM51Config,
    KimiK25Config,
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
    "KimiK25Config",
    "GLM51Config",
    "load_model_config",
    "MoEParallelStrategy",
    "HardwareMappingRequest",
    "HardwareMappingResult",
    "OperatorHardwareMapping",
    "HardwareMapper",
]
