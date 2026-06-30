# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import regex
import torch
from transformers import AutoModelForCausalLM
from vllm.logger import init_logger

from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import (
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator_multi,
)
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def _resolve_gguf_model_type(
    model_type: str,
    aliases: tuple[str, ...],
) -> str:
    supported_model_types = set(gguf.MODEL_ARCH_NAMES.values())
    for alias in aliases:
        if alias in supported_model_types:
            return alias
    return model_type


def _sanitize_deepseek_v4_rope_parameters(params: dict) -> dict:
    sanitized = {}
    for key, value in params.items():
        if key in ("main", "compress"):
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            continue
        sanitized["rope_type" if key == "type" else key] = value
    return sanitized


def _normalize_deepseek_v4_config(config: "PretrainedConfig") -> None:
    if config.model_type not in ("deepseek_v4", "deepseek_v4_flash"):
        return

    missing = object()

    def get_attr(obj: "PretrainedConfig", attr: str):
        try:
            return getattr(obj, attr)
        except AttributeError:
            return missing

    def ensure(obj: "PretrainedConfig", attr: str, value) -> None:
        current = get_attr(obj, attr)
        if current is missing or (current is None and value is not None):
            obj.update({attr: value})

    scalar_defaults = {
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "num_experts_per_tok": 6,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1.0e-6,
        "swiglu_limit": 10.0,
        "sliding_window": 128,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "num_nextn_predict_layers": 1,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1.0e-6,
        "use_cache": True,
        "pad_token_id": None,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
    }
    for attr, value in scalar_defaults.items():
        ensure(config, attr, value)
    ensure(config, "intermediate_size", config.moe_intermediate_size)
    ensure(config, "num_local_experts", config.n_routed_experts)

    compress_rope_theta = get_attr(config, "compress_rope_theta")
    if compress_rope_theta is missing or compress_rope_theta is None:
        config.update({"compress_rope_theta": 160000.0})

    partial_rotary_factor = get_attr(config, "partial_rotary_factor")
    if partial_rotary_factor is missing or partial_rotary_factor is None:
        qk_rope_head_dim = get_attr(config, "qk_rope_head_dim")
        partial_rotary_factor = (
            qk_rope_head_dim / config.head_dim
            if qk_rope_head_dim is not missing and qk_rope_head_dim is not None
            else 64 / 512
        )
        config.update({"partial_rotary_factor": partial_rotary_factor})

    n_layers = config.num_hidden_layers
    compress_rates = get_attr(config, "compress_rates")
    if compress_rates is missing or compress_rates is None:
        compress_rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
    else:
        compress_rates = dict(compress_rates)
    legacy_compress_rate_csa = get_attr(config, "compress_rate_csa")
    if (
        legacy_compress_rate_csa is not missing
        and legacy_compress_rate_csa is not None
    ):
        compress_rates["compressed_sparse_attention"] = legacy_compress_rate_csa
    legacy_compress_rate_hca = get_attr(config, "compress_rate_hca")
    if (
        legacy_compress_rate_hca is not missing
        and legacy_compress_rate_hca is not None
    ):
        compress_rates["heavily_compressed_attention"] = legacy_compress_rate_hca
    config.update({"compress_rates": compress_rates})

    compress_ratio_to_layer_type = {
        0: "sliding_attention",
        4: "compressed_sparse_attention",
        128: "heavily_compressed_attention",
    }
    layer_types = get_attr(config, "layer_types")
    if layer_types is missing or layer_types is None:
        legacy_compress_ratios = get_attr(config, "compress_ratios")
        if (
            legacy_compress_ratios is not missing
            and legacy_compress_ratios is not None
        ):
            layer_types = [
                compress_ratio_to_layer_type[r] for r in legacy_compress_ratios
            ]
        else:
            layer_types = ["heavily_compressed_attention"] * min(n_layers, 2) + [
                "compressed_sparse_attention"
                if i % 2
                else "heavily_compressed_attention"
                for i in range(max(n_layers - 2, 0))
            ]
    config.update({"layer_types": list(layer_types[:n_layers])})

    mlp_layer_types = get_attr(config, "mlp_layer_types")
    if mlp_layer_types is missing or mlp_layer_types is None:
        legacy_num_hash_layers = get_attr(config, "num_hash_layers")
        n_hash = (
            legacy_num_hash_layers
            if (
                legacy_num_hash_layers is not missing
                and legacy_num_hash_layers is not None
            )
            else 3
        )
        mlp_layer_types = ["hash_moe"] * min(n_layers, n_hash) + [
            "moe" for _ in range(max(n_layers - n_hash, 0))
        ]
    config.update({"mlp_layer_types": list(mlp_layer_types[:n_layers])})

    ensure(
        config,
        "qk_rope_head_dim",
        int(config.head_dim * config.partial_rotary_factor),
    )
    rope_parameters = get_attr(config, "rope_parameters")
    if rope_parameters is missing or rope_parameters is None:
        rope_parameters = {}
    if (
        isinstance(rope_parameters.get("main"), dict)
        and isinstance(rope_parameters.get("compress"), dict)
    ):
        main = _sanitize_deepseek_v4_rope_parameters(rope_parameters["main"])
        compress = _sanitize_deepseek_v4_rope_parameters(
            rope_parameters["compress"]
        )
        main.setdefault("rope_type", "default")
        main["rope_theta"] = config.rope_theta
        main["partial_rotary_factor"] = config.partial_rotary_factor
        compress.setdefault("rope_type", "default")
        compress["rope_theta"] = config.compress_rope_theta
        compress["partial_rotary_factor"] = config.partial_rotary_factor
        if compress["rope_type"] == "yarn":
            compress.setdefault("attention_factor", 1.0)
        rope_parameters = {
            "main": main,
            "compress": compress,
        }
    else:
        yarn = _sanitize_deepseek_v4_rope_parameters(rope_parameters)
        main = {
            "rope_type": "default",
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": config.partial_rotary_factor,
        }
        compress = {
            **yarn,
            "rope_theta": config.compress_rope_theta,
            "partial_rotary_factor": config.partial_rotary_factor,
        }
        compress.setdefault("rope_type", "default")
        if compress["rope_type"] == "yarn":
            compress.setdefault("attention_factor", 1.0)
        rope_parameters = {"main": main, "compress": compress}
    config.update({"rope_parameters": rope_parameters})

    text_config = config.get_text_config()
    if text_config is not config:
        _normalize_deepseek_v4_config(text_config)


class GGUFWeightsAdapter(BaseGGUFWeightsAdapter):
    """Default adapter for GGUF models."""

    load_spec = None

    @classmethod
    def matches(cls, config) -> bool:
        del config
        return True

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        _normalize_deepseek_v4_config(hf_config)
        hf_config = maybe_patch_hf_config_from_gguf(model_path, hf_config)
        _normalize_deepseek_v4_config(hf_config)
        return hf_config

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = (
            hasattr(config, "vision_config") and config.vision_config is not None
        )

        gguf_to_hf_name_map: dict[str, str] = {}
        sideload_params: list[re.Pattern] = []

        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            model_type = "gemma3"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type in ("deepseek_v4", "deepseek_v4_flash"):
            model_type = _resolve_gguf_model_type(
                model_type,
                ("deepseek4", "deepseek2"),
            )
            gguf_to_hf_name_map["output_hc_fn.weight"] = "model.hc_head_fn"
            gguf_to_hf_name_map["output_hc_base.weight"] = "model.hc_head_base"
            gguf_to_hf_name_map["output_hc_scale.weight"] = "model.hc_head_scale"
            sideload_params.extend(
                regex.compile(pattern)
                for pattern in (
                    r"model\.hc_head\.(hc_fn|hc_base|hc_scale)",
                    r"model\.layers\.[0-9]+\.self_attn\."
                    r"(q_a_norm|kv_proj|kv_norm|o_a_proj|o_b_proj)\.weight",
                    r"model\.layers\.[0-9]+\.self_attn\.compressor\."
                    r"(position_bias|kv_proj\.weight|gate_proj\.weight|"
                    r"kv_norm\.weight)",
                    r"model\.layers\.[0-9]+\.self_attn\.compressor\.indexer\."
                    r"(position_bias|kv_proj\.weight|gate_proj\.weight|"
                    r"kv_norm\.weight|q_b_proj\.weight|"
                    r"scorer\.weights_proj\.weight)",
                    r"model\.layers\.[0-9]+\.mlp\.gate\."
                    r"(tid2eid|e_score_correction_bias)",
                    r"model\.layers\.[0-9]+\.mlp\.experts\."
                    r"(gate_up_proj|down_proj)",
                    r"model\.layers\.[0-9]+\.mlp\.shared_experts\."
                    r"(gate_proj|up_proj|down_proj)\.weight",
                    r"model\.layers\.[0-9]+\.(attn_hc|ffn_hc)\."
                    r"(fn|base|scale)",
                )
            )
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.attn_norm.weight"] = (
                    f"model.layers.{idx}.attn_norm.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_sinks.weight"] = (
                    f"model.layers.{idx}.attn.attn_sink"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_q_a.weight"] = (
                    f"model.layers.{idx}.attn.wq_a.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_q_b.weight"] = (
                    f"model.layers.{idx}.attn.wq_b.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_q_a_norm.weight"] = (
                    f"model.layers.{idx}.attn.q_norm.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_kv.weight"] = (
                    f"model.layers.{idx}.attn.wkv.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_kv_a_norm.weight"] = (
                    f"model.layers.{idx}.attn.kv_norm.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_output_a.weight"] = (
                    f"model.layers.{idx}.attn.wo_a.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_output_b.weight"] = (
                    f"model.layers.{idx}.attn.wo_b.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.hc_attn_fn.weight"] = (
                    f"model.layers.{idx}.hc_attn_fn"
                )
                gguf_to_hf_name_map[f"blk.{idx}.hc_attn_base.weight"] = (
                    f"model.layers.{idx}.hc_attn_base"
                )
                gguf_to_hf_name_map[f"blk.{idx}.hc_attn_scale.weight"] = (
                    f"model.layers.{idx}.hc_attn_scale"
                )
                gguf_to_hf_name_map[f"blk.{idx}.hc_ffn_fn.weight"] = (
                    f"model.layers.{idx}.hc_ffn_fn"
                )
                gguf_to_hf_name_map[f"blk.{idx}.hc_ffn_base.weight"] = (
                    f"model.layers.{idx}.hc_ffn_base"
                )
                gguf_to_hf_name_map[f"blk.{idx}.hc_ffn_scale.weight"] = (
                    f"model.layers.{idx}.hc_ffn_scale"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_compressor_ape.weight"] = (
                    f"model.layers.{idx}.attn.compressor.ape"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_compressor_kv.weight"] = (
                    f"model.layers.{idx}.attn.compressor.wkv.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_compressor_gate.weight"] = (
                    f"model.layers.{idx}.attn.compressor.wgate.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.attn_compressor_norm.weight"] = (
                    f"model.layers.{idx}.attn.compressor.norm.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.indexer_compressor_ape.weight"] = (
                    f"model.layers.{idx}.attn.indexer.compressor.ape"
                )
                gguf_to_hf_name_map[f"blk.{idx}.indexer_compressor_kv.weight"] = (
                    f"model.layers.{idx}.attn.indexer.compressor.wkv.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.indexer_compressor_gate.weight"] = (
                    f"model.layers.{idx}.attn.indexer.compressor.wgate.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.indexer_compressor_norm.weight"] = (
                    f"model.layers.{idx}.attn.indexer.compressor.norm.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.indexer.attn_q_b.weight"] = (
                    f"model.layers.{idx}.attn.indexer.wq_b.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.indexer.proj.weight"] = (
                    f"model.layers.{idx}.attn.indexer.weights_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_tid2eid.weight"] = (
                    f"model.layers.{idx}.ffn.gate.tid2eid"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_inp.weight"] = (
                    f"model.layers.{idx}.ffn.gate.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.ffn.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_norm.weight"] = (
                    f"model.layers.{idx}.ffn_norm.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.ffn.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.ffn.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.ffn.experts.0.w3.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_shexp.weight"] = (
                    f"model.layers.{idx}.ffn.shared_experts.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_shexp.weight"] = (
                    f"model.layers.{idx}.ffn.shared_experts.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_shexp.weight"] = (
                    f"model.layers.{idx}.ffn.shared_experts.w3.weight"
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "olmoe":
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.extend(
                    [
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                        ),
                        regex.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.(gate_up_proj|down_proj)"
                        ),
                    ]
                )
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    regex.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        text_name_map = gguf.get_tensor_name_map(arch, text_config.num_hidden_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_name_map = gguf.get_tensor_name_map(
                mm_proj_arch, config.vision_config.num_hidden_layers
            )
        else:
            vision_name_map = None

        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]
                if is_multimodal:
                    hf_name = "model." + hf_name
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]
                    suffix = "weight"
            gguf_name = None
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)
            if gguf_name is None:
                return None
            return gguf_name + "." + suffix

        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            if gguf_name_with_suffix is not None:
                # Preserve hardcoded mappings (e.g., DeepSeek V4's
                # blk.{idx}.attn_q_a.weight → model.layers.{idx}.attn.wq_a.weight
                # which the model's stacked_params_mapping expects).
                # Auto-mapping would override them with standard HF names
                # (e.g., ...self_attn.q_a_proj.weight) that don't match.
                if gguf_name_with_suffix not in gguf_to_hf_name_map:
                    gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                    logger.debug(
                        "Mapped GGUF %s → HF %s",
                        gguf_name_with_suffix,
                        hf_name,
                    )
                else:
                    logger.debug(
                        "Skipping GGUF %s → HF %s (already mapped to %s)",
                        gguf_name_with_suffix,
                        hf_name,
                        gguf_to_hf_name_map[gguf_name_with_suffix],
                    )
            elif hf_name not in gguf_to_hf_name_map.values():
                unmapped_params.append(hf_name)

        if unmapped_params:
            unmapped_params = [
                x
                for x in unmapped_params
                if not any(regex.fullmatch(p, x) for p in sideload_params)
            ]
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): {unmapped_params}"
            )
        return gguf_to_hf_name_map

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for hf_name, weight in weights:
            yield hf_name, self.transform_weight(hf_name, weight)

    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
        if not match:
            return [model_path]
        total = int(match.group(2))
        num_digits = len(match.group(1))
        prefix = model_path[: match.start(1)]
        suffix = model_path[match.end(2) :]
        files = []
        for i in range(1, total + 1):
            shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            if os.path.isfile(shard_path):
                files.append(shard_path)
        if files:
            logger.info("Discovered %d GGUF shard files", len(files))
        return files if files else [model_path]

    def update_tie_word_embeddings(
        self,
        model_path: str,
        hf_config: PretrainedConfig,
        gguf_to_hf_name_map: dict[str, str],
    ) -> None:
        if "lm_head.weight" not in gguf_to_hf_name_map.values():
            return

        all_extra_names = []
        for gguf_file in self._get_all_gguf_files(model_path):
            all_extra_names.extend(
                get_gguf_extra_tensor_names(gguf_file, gguf_to_hf_name_map)
            )
        hf_config.update({"tie_word_embeddings": "lm_head.weight" in all_extra_names})

    def get_weight_type_map(
        self,
        model_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map = {}
        for gguf_file in self._get_all_gguf_files(model_path):
            weight_type_map.update(
                get_gguf_weight_type_map(gguf_file, gguf_to_hf_name_map)
            )
        return weight_type_map

    @staticmethod
    def get_unquantized_modules(weight_type_map: dict[str, str]) -> list[str]:
        return [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        model_config.hf_config = self.patch_hf_config(
            model_path, model_config.hf_config
        )
        gguf_to_hf_name_map = self.build_name_map(model_config)
        self.update_tie_word_embeddings(
            model_path, model_config.hf_config, gguf_to_hf_name_map
        )
        weight_type_map = self.get_weight_type_map(model_path, gguf_to_hf_name_map)
        self.load_spec = GGUFLoadSpec(
            weights_source=self._get_all_gguf_files(model_path),
            gguf_to_hf_name_map=gguf_to_hf_name_map,
            unquantized_modules=self.get_unquantized_modules(weight_type_map),
        )
        return self.load_spec

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        weights = gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source,
            self.load_spec.gguf_to_hf_name_map,
        )
        yield from self.map_weights(weights)
