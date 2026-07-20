"""GGUF metadata and tokenizer export for BTL-3 Compact."""

from __future__ import annotations

import json
from pathlib import Path


def _tokenizer_payload(source: Path, vocab_size: int, gguf):
    payload = json.loads((source / "tokenizer.json").read_text())
    vocab = payload["model"]["vocab"]
    added = {int(item["id"]): item for item in payload.get("added_tokens", [])}
    reverse = {int(index): token for token, index in vocab.items()}
    tokens = []
    types = []
    for index in range(vocab_size):
        if index in reverse:
            tokens.append(reverse[index])
            types.append(gguf.TokenType.NORMAL)
        elif index in added:
            item = added[index]
            token = str(item["content"])
            tokens.append(token)
            looks_control = token.startswith("<|") and token.endswith("|>")
            types.append(
                gguf.TokenType.CONTROL
                if item.get("special") or looks_control
                else gguf.TokenType.USER_DEFINED
            )
        else:
            tokens.append(f"[PAD{index}]")
            types.append(gguf.TokenType.UNUSED)
    merges = payload["model"].get("merges", [])
    encoded_merges = []
    for merge in merges:
        if isinstance(merge, str):
            encoded_merges.append(merge)
        elif isinstance(merge, list) and len(merge) == 2:
            encoded_merges.append(
                " ".join(part.replace(" ", chr(ord(" ") + 256)) for part in merge)
            )
        else:
            raise ValueError(f"unsupported tokenizer merge: {merge!r}")
    return tokens, types, encoded_merges


def add_metadata(writer, gguf, source: Path, *, include_tokenizer: bool) -> None:
    config = json.loads((source / "config.json").read_text())
    rope = config["rope_parameters"]
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("BTL-3 Compact")
    writer.add_basename("BTL-3")
    writer.add_version("Compact")
    writer.add_description("BTL-3 Compact native AVQ2 text model")
    writer.add_license("apache-2.0")
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_key_length(config["head_dim"])
    writer.add_value_length(config["head_dim"])
    writer.add_rope_dimension_count(
        int(config["head_dim"] * config["partial_rotary_factor"])
    )
    sections = list(rope["mrope_section"])
    writer.add_rope_dimension_sections((sections + [0])[:4])
    writer.add_rope_freq_base(rope["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_full_attention_interval(config["full_attention_interval"])
    writer.add_ssm_conv_kernel(config["linear_conv_kernel_dim"])
    writer.add_ssm_inner_size(
        config["linear_value_head_dim"] * config["linear_num_value_heads"]
    )
    writer.add_ssm_state_size(config["linear_key_head_dim"])
    writer.add_ssm_time_step_rank(config["linear_num_value_heads"])
    writer.add_ssm_group_count(config["linear_num_key_heads"])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_bool("btl3.compact", True)
    writer.add_uint32("btl3.export.schema_version", 1)
    writer.add_string(
        "btl3.base.revision",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
    writer.add_float32("btl3.behavior_lora_scale", 2.0)
    if not include_tokenizer:
        writer.add_string("btl3.export.mode", "one-layer-conformance")
        return

    tokens, types, merges = _tokenizer_payload(source, config["vocab_size"], gguf)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_eos_token_id(248046)
    writer.add_pad_token_id(248044)
    writer.add_chat_template((source / "chat_template.jinja").read_text())
