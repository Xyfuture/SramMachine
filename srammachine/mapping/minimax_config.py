"""Validated text-decoder configuration from the MiniMax M3 model card."""

from dataclasses import dataclass
from typing import Any, Mapping

from .model import ModelConfig, _boolean, _exact, _positive_int, _positive_number


@dataclass(frozen=True)
class MiniMaxM3Config(ModelConfig):
    """M3 GQA/MSA parameters; inherited MLA-only dimensions remain zero."""

    num_key_value_heads: int = 4
    head_dim: int = 128
    sparse_topk_blocks: int = 16
    sparse_block_size: int = 128
    shared_intermediate_size: int = 3072
    n_shared_experts: int = 1
    sparse_attention_layers: tuple[bool, ...] = ()
    moe_layers: tuple[bool, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, model_name: str = "minimax-m3") -> "MiniMaxM3Config":
        """Read nested HF text_config and preserve its heterogeneous layer masks."""
        if not isinstance(data, Mapping):
            raise TypeError("model card must be a mapping")
        _exact(data, "model_type", "minimax_m3_vl")
        text = data["text_config"]
        if not isinstance(text, Mapping):
            raise TypeError("text_config must be a mapping")
        sparse = text["sparse_attention_config"]
        if not isinstance(sparse, Mapping):
            raise TypeError("sparse_attention_config must be a mapping")
        _exact(sparse, "use_sparse_attention", True)
        _exact(sparse, "sparse_score_type", "max")
        _exact(sparse, "sparse_init_block", 0)
        _exact(sparse, "sparse_local_block", 1)
        heads = _positive_int(text, "num_attention_heads")
        kv_heads = _positive_int(text, "num_key_value_heads")
        index_heads = _positive_int(sparse, "sparse_num_index_heads")
        if heads % kv_heads or index_heads != kv_heads:
            raise ValueError("M3 requires complete GQA groups and one index query head per KV head")
        layers = _positive_int(text, "num_hidden_layers")

        def mask(source, key):
            values = source[key]
            if (not isinstance(values, (list, tuple)) or len(values) != layers
                    or any(type(v) is not int or v not in (0, 1) for v in values)):
                raise ValueError(f"{key} must contain one 0/1 entry per layer")
            return tuple(bool(v) for v in values)

        sparse_layers = mask(sparse, "sparse_attention_freq")
        moe_layers = mask(text, "moe_layer_freq")
        disabled_values = mask(sparse, "sparse_disable_index_value")
        if any(active and not disabled for active, disabled in zip(sparse_layers, disabled_values)):
            raise ValueError("sparse index value heads are not supported")
        if sparse_layers != moe_layers or not any(sparse_layers) or all(sparse_layers):
            raise ValueError("M3 requires dense GQA/FFN and sparse MSA/MoE layer types")
        experts = _positive_int(text, "num_local_experts")
        top_k = _positive_int(text, "num_experts_per_tok")
        if top_k > experts:
            raise ValueError("num_experts_per_tok must not exceed expert count")
        return cls(
            model_name=model_name,
            hidden_size=_positive_int(text, "hidden_size"),
            intermediate_size=_positive_int(text, "dense_intermediate_size"),
            moe_intermediate_size=_positive_int(text, "intermediate_size"),
            num_attention_heads=heads, num_key_value_heads=kv_heads,
            head_dim=_positive_int(text, "head_dim"),
            num_experts=experts, top_k=top_k, num_hidden_layers=layers,
            num_nextn_predict_layers=_positive_int(text, "num_nextn_predict_layers"),
            max_position_embeddings=_positive_int(text, "max_position_embeddings"),
            q_lora_rank=0, kv_lora_rank=0, qk_nope_head_dim=0, qk_rope_head_dim=0,
            v_head_dim=_positive_int(text, "head_dim"),
            rms_norm_eps=_positive_number(text, "rms_norm_eps"),
            rope_theta=_positive_number(text, "rope_theta"),
            use_qk_norm=_boolean(text, "use_qk_norm"),
            indexer_num_heads=index_heads,
            indexer_head_dim=_positive_int(sparse, "sparse_index_dim"),
            sparse_topk_blocks=_positive_int(sparse, "sparse_topk_blocks"),
            sparse_block_size=_positive_int(sparse, "sparse_block_size"),
            shared_intermediate_size=_positive_int(text, "shared_intermediate_size"),
            n_shared_experts=_positive_int(text, "n_shared_experts"),
            sparse_attention_layers=sparse_layers, moe_layers=moe_layers,
        )
