#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace btl3 {

struct avq2_matrix {
    const uint8_t * codes;
    const float * affine_weight;
    const float * affine_bias;
    const int8_t * input_signs;
    const int8_t * output_signs;
    std::size_t rows;
    std::size_t columns;
    std::size_t codebook_rows;
    std::size_t block_size;
};

std::array<float, 4> decode_avq2_code(uint8_t code);

void hadamard_blocks(std::vector<float> & values, std::size_t block_size);

void mul_mat_vec(const avq2_matrix & matrix, const float * input, float * output);

ggml_tensor * build_mul_mat(
    ggml_context * ctx,
    ggml_tensor * codes,
    ggml_tensor * affine_weight,
    ggml_tensor * affine_bias,
    ggml_tensor * input_signs,
    ggml_tensor * output_signs,
    ggml_tensor * input);

ggml_tensor * build_int4_mul_mat(
    ggml_context * ctx,
    ggml_tensor * codes,
    ggml_tensor * scales,
    ggml_tensor * zeros,
    ggml_tensor * input);

ggml_tensor * build_vocab_get_rows(
    ggml_context * ctx,
    ggml_tensor * codes,
    ggml_tensor * affine_weight,
    ggml_tensor * affine_bias,
    ggml_tensor * input_signs,
    ggml_tensor * rescued_row_ids,
    ggml_tensor * rescued_upper_six,
    ggml_tensor * rescued_scales,
    ggml_tensor * token_ids);

ggml_tensor * build_vocab_head(
    ggml_context * ctx,
    ggml_tensor * codes,
    ggml_tensor * affine_weight,
    ggml_tensor * affine_bias,
    ggml_tensor * input_signs,
    ggml_tensor * input);

} // namespace btl3
