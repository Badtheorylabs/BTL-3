#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace btl3 {

constexpr std::size_t avq2_components_per_code = 4;
constexpr std::size_t avq2_transform_block = 128;
constexpr std::uintptr_t avq2_custom_magic =
    UINT64_C(0x42544c3341565132);
constexpr std::uintptr_t int4_custom_magic =
    UINT64_C(0x42544c33494e5434);
constexpr std::uintptr_t vocab_rows_custom_magic =
    UINT64_C(0x42544c3356475257);
constexpr std::uintptr_t vocab_head_custom_magic =
    UINT64_C(0x42544c3356484544);
constexpr std::uintptr_t packed_probe_custom_magic =
    UINT64_C(0x42544c3350524f42);
constexpr std::uintptr_t metal_packed_probe_custom_magic =
    UINT64_C(0x42544c334d505242);
constexpr std::uintptr_t cuda_avq2_probe_custom_magic =
    UINT64_C(0x42544c3343415651);
constexpr std::uintptr_t cuda_int4_probe_custom_magic =
    UINT64_C(0x42544c3343494e54);
constexpr std::uintptr_t cuda_vocab_rows_probe_custom_magic =
    UINT64_C(0x42544c3343565257);
constexpr std::uintptr_t cuda_vocab_head_probe_custom_magic =
    UINT64_C(0x42544c3343564844);

struct avq2_shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t codebook_rows;
    std::size_t block_size;
};

enum class avq2_shape_error {
    none,
    empty,
    packed_columns,
    codebook_rows,
    transform_block,
};

constexpr avq2_shape_error validate_avq2_shape(avq2_shape shape) {
    if (shape.rows == 0 || shape.columns == 0) {
        return avq2_shape_error::empty;
    }
    if (shape.columns % avq2_components_per_code != 0) {
        return avq2_shape_error::packed_columns;
    }
    if (shape.codebook_rows == 0 ||
        shape.rows % shape.codebook_rows != 0) {
        return avq2_shape_error::codebook_rows;
    }
    if (shape.block_size == 0 ||
        (shape.block_size & (shape.block_size - 1)) != 0 ||
        shape.rows % shape.block_size != 0 ||
        shape.columns % shape.block_size != 0) {
        return avq2_shape_error::transform_block;
    }
    return avq2_shape_error::none;
}

constexpr std::size_t avq2_workspace_bytes(
        avq2_shape shape,
        std::size_t token_count) {
    if (validate_avq2_shape(shape) != avq2_shape_error::none ||
        token_count == 0) {
        return 0;
    }
    constexpr std::size_t limit =
        std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (shape.columns > limit - shape.rows) {
        return 0;
    }
    const std::size_t values_per_token = shape.columns + shape.rows;
    if (token_count > limit / values_per_token) {
        return 0;
    }
    return token_count * values_per_token * sizeof(float);
}

} // namespace btl3
