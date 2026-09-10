"""Validated model cards supported by the decode hardware mapper."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


_MODEL_CARD_DIR = Path(__file__).resolve().parent.parent / "model_cards"
_SUPPORTED_MODELS = {
    "deepseek-v3": "deepseek-v3.json",
    "deepseek-v3.2": "deepseek-v3.2.json",
}


def _positive_int(data: Mapping[str, Any], name: str) -> int:
    if name not in data:
        raise KeyError(f"model card missing required field: {name}")
    value = data[name]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(data: Mapping[str, Any], name: str) -> float:
    if name not in data:
        raise KeyError(f"model card missing required field: {name}")
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _exact(data: Mapping[str, Any], name: str, expected: Any) -> Any:
    if name not in data:
        raise KeyError(f"model card missing required field: {name}")
    value = data[name]
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"unsupported {name}: {value!r}")
    return value


def _boolean(data: Mapping[str, Any], name: str) -> bool:
    if name not in data:
        raise KeyError(f"model card missing required field: {name}")
    value = data[name]
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class ModelConfig:
    """Common validated configuration consumed by the hardware mapper."""

    model_name: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_attention_heads: int
    num_experts: int
    top_k: int
    num_hidden_layers: int
    max_position_embeddings: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    rope_theta: float
    use_qk_norm: bool
    dsa: bool = False
    dsa_len: int | None = None
    indexer_num_heads: int | None = None
    indexer_head_dim: int | None = None

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, model_name: str = "deepseek-v3",
    ) -> "ModelConfig":
        if not isinstance(data, Mapping):
            raise TypeError("model card must be a mapping")
        _exact(data, "attn_type", "mla")
        _exact(data, "ffn_type", "moe")
        _exact(data, "hidden_act", "silu")
        _exact(data, "attention_bias", False)
        _exact(data, "dsa", False)
        _exact(data, "num_nextn_predict_layers", 1)
        heads = _positive_int(data, "num_attention_heads")
        hidden = _positive_int(data, "hidden_size")
        experts = _positive_int(data, "n_routed_experts")
        top_k = _positive_int(data, "num_experts_per_tok")
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if top_k > experts:
            raise ValueError("num_experts_per_tok must not exceed routed experts")
        return cls(
            model_name=model_name,
            hidden_size=hidden,
            intermediate_size=_positive_int(data, "intermediate_size"),
            moe_intermediate_size=_positive_int(data, "moe_intermediate_size"),
            num_attention_heads=heads,
            num_experts=experts,
            top_k=top_k,
            num_hidden_layers=_positive_int(data, "num_hidden_layers"),
            max_position_embeddings=_positive_int(data, "max_position_embeddings"),
            q_lora_rank=_positive_int(data, "q_lora_rank"),
            kv_lora_rank=_positive_int(data, "kv_lora_rank"),
            qk_nope_head_dim=_positive_int(data, "qk_nope_head_dim"),
            qk_rope_head_dim=_positive_int(data, "qk_rope_head_dim"),
            v_head_dim=_positive_int(data, "v_head_dim"),
            rms_norm_eps=_positive_number(data, "rms_norm_eps"),
            rope_theta=_positive_number(data, "rope_theta"),
            use_qk_norm=_boolean(data, "use_qk_norm"),
        )


@dataclass(frozen=True)
class DeepSeekV3Config(ModelConfig):
    """DeepSeek V3 MLA/MoE layer parameters."""


@dataclass(frozen=True)
class DeepSeekV32Config(DeepSeekV3Config):
    """DeepSeek V3.2 parameters, including the sparse-attention indexer."""

    dsa: bool = True

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, model_name: str = "deepseek-v3.2",
    ) -> "DeepSeekV32Config":
        if not isinstance(data, Mapping):
            raise TypeError("model card must be a mapping")
        _exact(data, "attn_type", "mla")
        _exact(data, "ffn_type", "moe")
        _exact(data, "hidden_act", "silu")
        _exact(data, "attention_bias", False)
        _exact(data, "dsa", True)
        _exact(data, "topk_sharing", False)
        _exact(data, "num_nextn_predict_layers", 1)
        heads = _positive_int(data, "num_attention_heads")
        hidden = _positive_int(data, "hidden_size")
        experts = _positive_int(data, "n_routed_experts")
        top_k = _positive_int(data, "num_experts_per_tok")
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if top_k > experts:
            raise ValueError("num_experts_per_tok must not exceed routed experts")
        return cls(
            model_name=model_name,
            hidden_size=hidden,
            intermediate_size=_positive_int(data, "intermediate_size"),
            moe_intermediate_size=_positive_int(data, "moe_intermediate_size"),
            num_attention_heads=heads,
            num_experts=experts,
            top_k=top_k,
            num_hidden_layers=_positive_int(data, "num_hidden_layers"),
            max_position_embeddings=_positive_int(data, "max_position_embeddings"),
            q_lora_rank=_positive_int(data, "q_lora_rank"),
            kv_lora_rank=_positive_int(data, "kv_lora_rank"),
            qk_nope_head_dim=_positive_int(data, "qk_nope_head_dim"),
            qk_rope_head_dim=_positive_int(data, "qk_rope_head_dim"),
            v_head_dim=_positive_int(data, "v_head_dim"),
            rms_norm_eps=_positive_number(data, "rms_norm_eps"),
            rope_theta=_positive_number(data, "rope_theta"),
            use_qk_norm=_boolean(data, "use_qk_norm"),
            dsa=True,
            dsa_len=_positive_int(data, "dsa_len"),
            indexer_num_heads=_positive_int(data, "indexer_num_heads"),
            indexer_head_dim=_positive_int(data, "indexer_head_dim"),
        )

def load_model_config(model_name: str) -> ModelConfig:
    """Load one of the two explicitly supported model cards."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")
    try:
        filename = _SUPPORTED_MODELS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported model: {model_name}") from error
    with (_MODEL_CARD_DIR / filename).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if model_name == "deepseek-v3":
        return DeepSeekV3Config.from_dict(data)
    return DeepSeekV32Config.from_dict(data)


__all__ = [
    "ModelConfig", "DeepSeekV3Config", "DeepSeekV32Config",
    "load_model_config",
]
