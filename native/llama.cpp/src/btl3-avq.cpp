#include "btl3-avq.h"
#include "btl3-avq-contract.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace btl3 {

std::array<float, 4> decode_avq2_code(uint8_t code) {
    return {
        static_cast<float>((code >> 6) & 3U) - 1.5F,
        static_cast<float>((code >> 4) & 3U) - 1.5F,
        static_cast<float>((code >> 2) & 3U) - 1.5F,
        static_cast<float>(code & 3U) - 1.5F,
    };
}

void hadamard_blocks(std::vector<float> & values, std::size_t block_size) {
    if (block_size == 0 || (block_size & (block_size - 1)) != 0) {
        throw std::invalid_argument("block size must be a positive power of two");
    }
    if (values.size() % block_size != 0) {
        throw std::invalid_argument("value count must be block aligned");
    }

    for (std::size_t start = 0; start < values.size(); start += block_size) {
        for (std::size_t width = 1; width < block_size; width *= 2) {
            for (std::size_t offset = 0; offset < block_size; offset += 2 * width) {
                for (std::size_t index = 0; index < width; ++index) {
                    const std::size_t left_index = start + offset + index;
                    const std::size_t right_index = left_index + width;
                    const float left = values[left_index];
                    const float right = values[right_index];
                    values[left_index] = left + right;
                    values[right_index] = left - right;
                }
            }
        }
    }

    const float inverse_scale = 1.0F / std::sqrt(static_cast<float>(block_size));
    for (float & value : values) {
        value *= inverse_scale;
    }
}

static void validate(const avq2_matrix & matrix) {
    if (matrix.codes == nullptr || matrix.affine_weight == nullptr ||
        matrix.affine_bias == nullptr || matrix.input_signs == nullptr ||
        matrix.output_signs == nullptr) {
        throw std::invalid_argument("AVQ2 matrix pointers must not be null");
    }
    const avq2_shape shape = {
        matrix.rows,
        matrix.columns,
        matrix.codebook_rows,
        matrix.block_size,
    };
    if (validate_avq2_shape(shape) != avq2_shape_error::none) {
        throw std::invalid_argument("AVQ2 matrix shape is invalid");
    }
}

void mul_mat_vec(const avq2_matrix & matrix, const float * input, float * output) {
    validate(matrix);
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("AVQ2 input and output must not be null");
    }

    std::vector<float> transformed_input(matrix.columns);
    for (std::size_t column = 0; column < matrix.columns; ++column) {
        transformed_input[column] =
            input[column] * static_cast<float>(matrix.input_signs[column]);
    }
    hadamard_blocks(transformed_input, matrix.block_size);

    std::vector<float> transformed_output(matrix.rows, 0.0F);
    const std::size_t codes_per_row = matrix.columns / 4;
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        const std::size_t group = row / matrix.codebook_rows;
        const float * group_weight = matrix.affine_weight + group * 16;
        const float * group_bias = matrix.affine_bias + group * 4;
        float accumulator = 0.0F;

        for (std::size_t code_index = 0; code_index < codes_per_row; ++code_index) {
            const auto lattice = decode_avq2_code(
                matrix.codes[row * codes_per_row + code_index]);
            for (std::size_t component = 0; component < 4; ++component) {
                float weight = group_bias[component];
                for (std::size_t lattice_index = 0; lattice_index < 4; ++lattice_index) {
                    weight += lattice[lattice_index] *
                              group_weight[component * 4 + lattice_index];
                }
                accumulator += weight * transformed_input[code_index * 4 + component];
            }
        }
        transformed_output[row] = accumulator;
    }

    hadamard_blocks(transformed_output, matrix.block_size);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        output[row] =
            transformed_output[row] * static_cast<float>(matrix.output_signs[row]);
    }
}

static void compute_mul_mat(
        ggml_tensor * dst,
        int           ith,
        int           nth,
        void        *) {
    const ggml_tensor * codes          = dst->src[0];
    const ggml_tensor * affine_weight  = dst->src[1];
    const ggml_tensor * affine_bias    = dst->src[2];
    const ggml_tensor * input_signs    = dst->src[3];
    const ggml_tensor * output_signs   = dst->src[4];
    const ggml_tensor * input          = dst->src[5];

    const std::size_t rows = static_cast<std::size_t>(dst->ne[0]);
    const std::size_t columns = static_cast<std::size_t>(input->ne[0]);
    const std::size_t groups = static_cast<std::size_t>(affine_bias->ne[1]);
    const std::size_t token_count =
        static_cast<std::size_t>(input->ne[1] * input->ne[2] * input->ne[3]);

    const avq2_matrix matrix = {
        static_cast<const uint8_t *>(codes->data),
        static_cast<const float *>(affine_weight->data),
        static_cast<const float *>(affine_bias->data),
        static_cast<const int8_t *>(input_signs->data),
        static_cast<const int8_t *>(output_signs->data),
        rows,
        columns,
        rows / groups,
        128,
    };

    const auto * input_data = static_cast<const float *>(input->data);
    auto * output_data = static_cast<float *>(dst->data);
    for (std::size_t token = static_cast<std::size_t>(ith);
            token < token_count;
            token += static_cast<std::size_t>(nth)) {
        mul_mat_vec(
            matrix,
            input_data + token * columns,
            output_data + token * rows);
    }
}

ggml_tensor * build_mul_mat(
        ggml_context * ctx,
        ggml_tensor * codes,
        ggml_tensor * affine_weight,
        ggml_tensor * affine_bias,
        ggml_tensor * input_signs,
        ggml_tensor * output_signs,
        ggml_tensor * input) {
    if (ctx == nullptr || codes == nullptr || affine_weight == nullptr ||
        affine_bias == nullptr || input_signs == nullptr ||
        output_signs == nullptr || input == nullptr) {
        throw std::invalid_argument("BTL3 AVQ2 graph tensors must not be null");
    }
    if (codes->type != GGML_TYPE_I8 || input_signs->type != GGML_TYPE_I8 ||
        output_signs->type != GGML_TYPE_I8 ||
        affine_weight->type != GGML_TYPE_F32 ||
        affine_bias->type != GGML_TYPE_F32 ||
        input->type != GGML_TYPE_F32) {
        throw std::invalid_argument("BTL3 AVQ2 graph tensor type is invalid");
    }
    if (!ggml_is_contiguous(codes) || !ggml_is_contiguous(affine_weight) ||
        !ggml_is_contiguous(affine_bias) || !ggml_is_contiguous(input_signs) ||
        !ggml_is_contiguous(output_signs) || !ggml_is_contiguous(input)) {
        throw std::invalid_argument("BTL3 AVQ2 graph tensors must be contiguous");
    }

    const int64_t columns = input->ne[0];
    const int64_t rows = output_signs->ne[0];
    const int64_t groups = affine_bias->ne[1];
    if (columns <= 0 || rows <= 0 || groups <= 0 ||
        columns % 128 != 0 || rows % 128 != 0 ||
        rows % groups != 0 ||
        codes->ne[0] != columns / 4 || codes->ne[1] != rows ||
        input_signs->ne[0] != columns ||
        affine_weight->ne[0] != 4 || affine_weight->ne[1] != 4 ||
        affine_weight->ne[2] != groups ||
        affine_bias->ne[0] != 4) {
        throw std::invalid_argument("BTL3 AVQ2 graph tensor shape is invalid");
    }

    ggml_tensor * args[] = {
        codes,
        affine_weight,
        affine_bias,
        input_signs,
        output_signs,
        input,
    };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        rows,
        input->ne[1],
        input->ne[2],
        input->ne[3],
        args,
        6,
        compute_mul_mat,
        GGML_N_TASKS_MAX,
        reinterpret_cast<void *>(avq2_custom_magic));
}

static void compute_int4_mul_mat(
        ggml_tensor * dst,
        int           ith,
        int           nth,
        void        *) {
    const ggml_tensor * codes  = dst->src[0];
    const ggml_tensor * scales = dst->src[1];
    const ggml_tensor * zeros  = dst->src[2];
    const ggml_tensor * input  = dst->src[3];
    const auto * code_data = static_cast<const uint8_t *>(codes->data);
    const auto * scale_data = static_cast<const ggml_fp16_t *>(scales->data);
    const auto * zero_data = static_cast<const uint8_t *>(zeros->data);
    const auto * input_data = static_cast<const float *>(input->data);
    auto * output_data = static_cast<float *>(dst->data);

    const std::size_t rows = static_cast<std::size_t>(dst->ne[0]);
    const std::size_t columns = static_cast<std::size_t>(input->ne[0]);
    const std::size_t groups_per_row = columns / 128;
    const std::size_t tokens =
        static_cast<std::size_t>(input->ne[1] * input->ne[2] * input->ne[3]);
    for (std::size_t token = static_cast<std::size_t>(ith);
            token < tokens;
            token += static_cast<std::size_t>(nth)) {
        for (std::size_t row = 0; row < rows; ++row) {
            float accumulator = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                const uint8_t packed =
                    code_data[row * (columns / 2) + column / 2];
                const uint8_t code =
                    column % 2 == 0 ? packed & 15U : packed >> 4;
                const std::size_t group =
                    row * groups_per_row + column / 128;
                const float scale = ggml_fp16_to_fp32(scale_data[group]);
                const float zero = static_cast<float>(zero_data[group]);
                accumulator +=
                    (static_cast<float>(code) - zero) * scale *
                    input_data[token * columns + column];
            }
            output_data[token * rows + row] = accumulator;
        }
    }
}

ggml_tensor * build_int4_mul_mat(
        ggml_context * ctx,
        ggml_tensor * codes,
        ggml_tensor * scales,
        ggml_tensor * zeros,
        ggml_tensor * input) {
    if (ctx == nullptr || codes == nullptr || scales == nullptr ||
        zeros == nullptr || input == nullptr) {
        throw std::invalid_argument("BTL3 INT4 graph tensors must not be null");
    }
    if (codes->type != GGML_TYPE_I8 || scales->type != GGML_TYPE_F16 ||
        zeros->type != GGML_TYPE_I8 || input->type != GGML_TYPE_F32) {
        throw std::invalid_argument("BTL3 INT4 graph tensor type is invalid");
    }
    const int64_t columns = input->ne[0];
    const int64_t rows = codes->ne[1];
    if (columns <= 0 || rows <= 0 || columns % 128 != 0 ||
        codes->ne[0] != columns / 2 ||
        scales->ne[0] != columns / 128 || scales->ne[1] != rows ||
        zeros->ne[0] != columns / 128 || zeros->ne[1] != rows) {
        throw std::invalid_argument("BTL3 INT4 graph tensor shape is invalid");
    }
    ggml_tensor * args[] = { codes, scales, zeros, input };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        rows,
        input->ne[1],
        input->ne[2],
        input->ne[3],
        args,
        4,
        compute_int4_mul_mat,
        GGML_N_TASKS_MAX,
        reinterpret_cast<void *>(int4_custom_magic));
}

} // namespace btl3
