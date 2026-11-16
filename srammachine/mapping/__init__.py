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
    LegacyHardwareMapper,
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
    "LegacyHardwareMapper",
]

from .model_mappers import (
    DeepSeekV3Mapper, DeepSeekV32Mapper, KimiK25Mapper, GLM51Mapper,
    mapper_for_model,
)

__all__ += ["DeepSeekV3Mapper", "DeepSeekV32Mapper", "KimiK25Mapper", "GLM51Mapper", "mapper_for_model"]

from .minimax import MiniMaxM3Mapper
from .minimax_config import MiniMaxM3Config

__all__ += ["MiniMaxM3Mapper", "MiniMaxM3Config"]

from .hbf_bandwidth import DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable

__all__ += ["DEFAULT_HBF_BANDWIDTH_CSV", "HBFBandwidthTable"]
