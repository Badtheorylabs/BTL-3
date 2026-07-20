#include "btl3-avq.h"
#include "btl3-avq-contract.h"

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace btl3 {
namespace {

float decode_component(
        const uint8_t * codes,
        const float * affine_weight,
        const float * affine_bias,
        std::size_t row,
        std::size_t column,
        std::size_t columns,
        std::size_t codebook_rows) {
    const auto lattice =
        decode_avq2_code(codes[row * (columns / 4) + column / 4]);
    const std::size_t component = column % 4;
    const std::size_t group = row / codebook_rows;
    const float * weight = affine_weight + group * 16;
    const float * bias = affine_bias + group * 4;
    float value = bias[component];
    for (std::size_t index = 0; index < 4; ++index) {
        value += lattice[index] * weight[component * 4 + index];
    }
    return value;
}

int find_rescue(const int32_t * row_ids, std::size_t count, int32_t token) {
    const int32_t * found = std::lower_bound(row_ids, row_ids + count, token);
    return found != row_ids + count && *found == token
        ? static_cast<int>(found - row_ids)
        : -1;
}

uint8_t unpack_upper_six(const uint8_t * packed, std::size_t index) {
    const std::size_t group = index / 4;
    const std::size_t lane = index % 4;
    const uint32_t word =
        static_cast<uint32_t>(packed[group * 3]) |
        (static_cast<uint32_t>(packed[group * 3 + 1]) << 8) |
        (static_cast<uint32_t>(packed[group * 3 + 2]) << 16);
    return static_cast<uint8_t>((word >> (lane * 6)) & 0x3FU);
}

void compute_vocab_get_rows(
        ggml_tensor * dst,
        int ith,
        int nth,
        void *) {
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * affine_weight = dst->src[1];
    const ggml_tensor * affine_bias = dst->src[2];
    const ggml_tensor * input_signs = dst->src[3];
    const ggml_tensor * rescued_row_ids = dst->src[4];
    const ggml_tensor * rescued_upper_six = dst->src[5];
    const ggml_tensor * rescued_scales = dst->src[6];
    const ggml_tensor * token_ids = dst->src[7];

    const auto * code_data = static_cast<const uint8_t *>(codes->data);
    const auto * weight_data = static_cast<const float *>(affine_weight->data);
    const auto * bias_data = static_cast<const float *>(affine_bias->data);
    const auto * sign_data = static_cast<const int8_t *>(input_signs->data);
    const auto * rescue_ids = static_cast<const int32_t *>(rescued_row_ids->data);
    const auto * upper_data =
        static_cast<const uint8_t *>(rescued_upper_six->data);
    const auto * scale_data =
        static_cast<const ggml_fp16_t *>(rescued_scales->data);
    const auto * ids = static_cast<const int32_t *>(token_ids->data);
    auto * output = static_cast<float *>(dst->data);

    const std::size_t columns = static_cast<std::size_t>(dst->ne[0]);
    const std::size_t tokens =
        static_cast<std::size_t>(dst->ne[1] * dst->ne[2] * dst->ne[3]);
    const std::size_t vocabulary = static_cast<std::size_t>(codes->ne[1]);
    const std::size_t groups = static_cast<std::size_t>(affine_bias->ne[1]);
    const std::size_t codebook_rows = vocabulary / groups;
    const std::size_t rescue_count =
        static_cast<std::size_t>(rescued_row_ids->ne[0]);
    const std::size_t upper_stride = columns * 3 / 4;

    for (std::size_t token_index = static_cast<std::size_t>(ith);
            token_index < tokens;
            token_index += static_cast<std::size_t>(nth)) {
        const int32_t token = ids[token_index];
        if (token < 0 || static_cast<std::size_t>(token) >= vocabulary) {
            continue;
        }
        float * row_output = output + token_index * columns;
        const int rescue = find_rescue(rescue_ids, rescue_count, token);
        if (rescue >= 0) {
            const uint8_t * low =
                code_data + static_cast<std::size_t>(token) * columns / 4;
            const uint8_t * upper =
                upper_data + static_cast<std::size_t>(rescue) * upper_stride;
            const float scale = ggml_fp16_to_fp32(scale_data[rescue]);
            for (std::size_t column = 0; column < columns; ++column) {
                const uint8_t low_two =
                    (low[column / 4] >> ((column % 4) * 2)) & 0x03U;
                const uint8_t packed =
                    low_two | (unpack_upper_six(upper, column) << 2);
                row_output[column] =
                    (static_cast<int>(packed) - 128) * scale;
            }
            continue;
        }
        for (std::size_t column = 0; column < columns; ++column) {
            row_output[column] = decode_component(
                code_data,
                weight_data,
                bias_data,
                static_cast<std::size_t>(token),
                column,
                columns,
                codebook_rows);
        }
        std::vector<float> transformed(row_output, row_output + columns);
        hadamard_blocks(transformed, 128);
        for (std::size_t column = 0; column < columns; ++column) {
            row_output[column] =
                transformed[column] * static_cast<float>(sign_data[column]);
        }
    }
}

void compute_vocab_head(
        ggml_tensor * dst,
        int ith,
        int nth,
        void *) {
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * affine_weight = dst->src[1];
    const ggml_tensor * affine_bias = dst->src[2];
    const ggml_tensor * input_signs = dst->src[3];
    const ggml_tensor * input = dst->src[4];

    const auto * code_data = static_cast<const uint8_t *>(codes->data);
    const auto * weight_data = static_cast<const float *>(affine_weight->data);
    const auto * bias_data = static_cast<const float *>(affine_bias->data);
    const auto * sign_data = static_cast<const int8_t *>(input_signs->data);
    const auto * input_data = static_cast<const float *>(input->data);
    auto * output = static_cast<float *>(dst->data);

    const std::size_t columns = static_cast<std::size_t>(input->ne[0]);
    const std::size_t vocabulary = static_cast<std::size_t>(dst->ne[0]);
    const std::size_t tokens =
        static_cast<std::size_t>(input->ne[1] * input->ne[2] * input->ne[3]);
    const std::size_t groups = static_cast<std::size_t>(affine_bias->ne[1]);
    const std::size_t codebook_rows = vocabulary / groups;

    for (std::size_t token = 0; token < tokens; ++token) {
        std::vector<float> transformed(columns);
        for (std::size_t column = 0; column < columns; ++column) {
            transformed[column] =
                input_data[token * columns + column] *
                static_cast<float>(sign_data[column]);
        }
        hadamard_blocks(transformed, 128);
        for (std::size_t row = static_cast<std::size_t>(ith);
                row < vocabulary;
                row += static_cast<std::size_t>(nth)) {
            float accumulator = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                accumulator += decode_component(
                    code_data,
                    weight_data,
                    bias_data,
                    row,
                    column,
                    columns,
                    codebook_rows) * transformed[column];
            }
            output[token * vocabulary + row] = accumulator;
        }
    }
}

void validate_vocab_base(
        ggml_tensor * codes,
        ggml_tensor * affine_weight,
        ggml_tensor * affine_bias,
        ggml_tensor * input_signs,
        int64_t columns) {
    if (codes == nullptr || affine_weight == nullptr ||
        affine_bias == nullptr || input_signs == nullptr ||
        codes->type != GGML_TYPE_I8 ||
        affine_weight->type != GGML_TYPE_F32 ||
        affine_bias->type != GGML_TYPE_F32 ||
        input_signs->type != GGML_TYPE_I8 ||
        columns <= 0 || columns % 128 != 0 ||
        codes->ne[0] != columns / 4 ||
        affine_weight->ne[0] != 4 || affine_weight->ne[1] != 4 ||
        affine_bias->ne[0] != 4 ||
        affine_weight->ne[2] != affine_bias->ne[1] ||
        codes->ne[1] % affine_bias->ne[1] != 0 ||
        input_signs->ne[0] != columns) {
        throw std::invalid_argument("BTL3 vocabulary tensor layout is invalid");
    }
}

} // namespace

ggml_tensor * build_vocab_get_rows(
        ggml_context * ctx,
        ggml_tensor * codes,
        ggml_tensor * affine_weight,
        ggml_tensor * affine_bias,
        ggml_tensor * input_signs,
        ggml_tensor * rescued_row_ids,
        ggml_tensor * rescued_upper_six,
        ggml_tensor * rescued_scales,
        ggml_tensor * token_ids) {
    if (ctx == nullptr || rescued_row_ids == nullptr ||
        rescued_upper_six == nullptr || rescued_scales == nullptr ||
        token_ids == nullptr) {
        throw std::invalid_argument("BTL3 embedding tensors must not be null");
    }
    const int64_t columns = input_signs->ne[0];
    validate_vocab_base(
        codes, affine_weight, affine_bias, input_signs, columns);
    if (rescued_row_ids->type != GGML_TYPE_I32 ||
        rescued_upper_six->type != GGML_TYPE_I8 ||
        rescued_scales->type != GGML_TYPE_F16 ||
        token_ids->type != GGML_TYPE_I32 ||
        rescued_upper_six->ne[0] != columns * 3 / 4 ||
        rescued_upper_six->ne[1] != rescued_row_ids->ne[0] ||
        rescued_scales->ne[0] != rescued_row_ids->ne[0]) {
        throw std::invalid_argument("BTL3 embedding rescue layout is invalid");
    }
    ggml_tensor * args[] = {
        codes,
        affine_weight,
        affine_bias,
        input_signs,
        rescued_row_ids,
        rescued_upper_six,
        rescued_scales,
        token_ids,
    };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        columns,
        token_ids->ne[0],
        token_ids->ne[1],
        token_ids->ne[2] * token_ids->ne[3],
        args,
        8,
        compute_vocab_get_rows,
        GGML_N_TASKS_MAX,
        reinterpret_cast<void *>(vocab_rows_custom_magic));
}

ggml_tensor * build_vocab_head(
        ggml_context * ctx,
        ggml_tensor * codes,
        ggml_tensor * affine_weight,
        ggml_tensor * affine_bias,
        ggml_tensor * input_signs,
        ggml_tensor * input) {
    if (ctx == nullptr || input == nullptr || input->type != GGML_TYPE_F32) {
        throw std::invalid_argument("BTL3 vocabulary head input is invalid");
    }
    validate_vocab_base(
        codes, affine_weight, affine_bias, input_signs, input->ne[0]);
    ggml_tensor * args[] = {
        codes,
        affine_weight,
        affine_bias,
        input_signs,
        input,
    };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        codes->ne[1],
        input->ne[1],
        input->ne[2],
        input->ne[3],
        args,
        5,
        compute_vocab_head,
        GGML_N_TASKS_MAX,
        reinterpret_cast<void *>(vocab_head_custom_magic));
}

} // namespace btl3
