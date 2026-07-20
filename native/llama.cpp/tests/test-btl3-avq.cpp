#include "../src/btl3-avq.h"
#include "../src/btl3-avq-contract.h"
#include "../ggml/src/ggml-cuda/btl3-avq-marker.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(float actual, float expected) {
    if (std::abs(actual - expected) > 1e-4F) {
        throw std::runtime_error(
            "BTL3 AVQ scalar mismatch: actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void custom_noop(ggml_tensor *, int, int, void *) {
}

} // namespace

int main() {
    constexpr btl3::avq2_shape valid_shape = {128, 256, 64, 128};
    static_assert(
        btl3::validate_avq2_shape(valid_shape) ==
        btl3::avq2_shape_error::none);
    static_assert(
        btl3::avq2_workspace_bytes(valid_shape, 2) ==
        2 * (128 + 256) * sizeof(float));
    static_assert(
        btl3::validate_avq2_shape({128, 255, 64, 128}) ==
        btl3::avq2_shape_error::packed_columns);
    static_assert(
        btl3::validate_avq2_shape({128, 256, 0, 128}) ==
        btl3::avq2_shape_error::codebook_rows);
    static_assert(
        btl3::validate_avq2_shape({128, 256, 64, 96}) ==
        btl3::avq2_shape_error::transform_block);
    const auto decoded = btl3::decode_avq2_code(UINT8_C(0x1B));
    const std::array<float, 4> expected = {-1.5F, -0.5F, 0.5F, 1.5F};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require_close(decoded[i], expected[i]);
    }

    std::vector<float> transformed = {1.0F, 2.0F, 3.0F, 4.0F};
    btl3::hadamard_blocks(transformed, 4);
    const std::array<float, 4> expected_h = {5.0F, -1.0F, -2.0F, 0.0F};
    for (std::size_t i = 0; i < expected_h.size(); ++i) {
        require_close(transformed[i], expected_h[i]);
    }

    const std::array<uint8_t, 4> codes = {0x1B, 0xE4, 0x00, 0xFF};
    const std::array<float, 16> affine = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    const std::array<float, 4> bias = {0.0F, 0.0F, 0.0F, 0.0F};
    const std::array<int8_t, 4> input_signs = {-1, 1, 1, -1};
    const std::array<int8_t, 4> output_signs = {1, -1, 1, -1};
    const std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 4> output = {};

    const btl3::avq2_matrix matrix = {
        codes.data(),
        affine.data(),
        bias.data(),
        input_signs.data(),
        output_signs.data(),
        4,
        4,
        4,
        4,
    };
    btl3::mul_mat_vec(matrix, input.data(), output.data());

    const std::array<float, 4> expected_output = {0.0F, 5.0F, 0.0F, 11.0F};
    for (std::size_t i = 0; i < expected_output.size(); ++i) {
        require_close(output[i], expected_output[i]);
    }

    ggml_init_params params = {
        1024 * 1024,
        nullptr,
        true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init(params),
        ggml_free);
    if (!ctx) {
        throw std::runtime_error("failed to create GGML test context");
    }

    ggml_tensor * graph_codes = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, 32, 128);
    ggml_tensor * graph_affine = ggml_new_tensor_3d(
        ctx.get(), GGML_TYPE_F32, 4, 4, 1);
    ggml_tensor * graph_bias = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, 4, 1);
    ggml_tensor * graph_input_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, 128);
    ggml_tensor * graph_output_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, 128);
    ggml_tensor * graph_input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, 128, 3);
    ggml_tensor * graph_output = btl3::build_mul_mat(
        ctx.get(),
        graph_codes,
        graph_affine,
        graph_bias,
        graph_input_signs,
        graph_output_signs,
        graph_input);
    if (graph_output->op != GGML_OP_CUSTOM ||
        graph_output->ne[0] != 128 ||
        graph_output->ne[1] != 3 ||
        !ggml_cuda_is_btl3_avq2(graph_output) ||
        GGML_CUDA_BTL3_AVQ2_MAGIC != btl3::avq2_custom_magic ||
        GGML_CUDA_BTL3_AVQ2_PROBE_MAGIC !=
            btl3::cuda_avq2_probe_custom_magic ||
        GGML_CUDA_BTL3_INT4_PROBE_MAGIC !=
            btl3::cuda_int4_probe_custom_magic ||
        GGML_CUDA_BTL3_VOCAB_ROWS_PROBE_MAGIC !=
            btl3::cuda_vocab_rows_probe_custom_magic ||
        GGML_CUDA_BTL3_VOCAB_HEAD_PROBE_MAGIC !=
            btl3::cuda_vocab_head_probe_custom_magic) {
        throw std::runtime_error("BTL3 AVQ2 graph shape mismatch");
    }
    ggml_tensor * marker_args[] = {graph_codes};
    auto make_probe = [&](std::uintptr_t magic) {
        return ggml_custom_4d(
            ctx.get(),
            GGML_TYPE_F32,
            1,
            1,
            1,
            1,
            marker_args,
            1,
            custom_noop,
            1,
            reinterpret_cast<void *>(magic));
    };
    ggml_tensor * avq2_probe = make_probe(
        btl3::cuda_avq2_probe_custom_magic);
    ggml_tensor * int4_probe = make_probe(
        btl3::cuda_int4_probe_custom_magic);
    ggml_tensor * vocab_rows_probe = make_probe(
        btl3::cuda_vocab_rows_probe_custom_magic);
    ggml_tensor * vocab_head_probe = make_probe(
        btl3::cuda_vocab_head_probe_custom_magic);
    ggml_tensor * unmarked = ggml_custom_4d(
        ctx.get(),
        GGML_TYPE_F32,
        1,
        1,
        1,
        1,
        marker_args,
        1,
        custom_noop,
        1,
        nullptr);
    if (!ggml_cuda_is_btl3_avq2_probe(avq2_probe) ||
        !ggml_cuda_is_btl3_int4_probe(int4_probe) ||
        !ggml_cuda_is_btl3_vocab_rows_probe(vocab_rows_probe) ||
        !ggml_cuda_is_btl3_vocab_head_probe(vocab_head_probe) ||
        ggml_cuda_is_btl3_avq2_probe(unmarked) ||
        ggml_cuda_is_btl3_int4_probe(unmarked) ||
        ggml_cuda_is_btl3_vocab_rows_probe(unmarked) ||
        ggml_cuda_is_btl3_vocab_head_probe(unmarked) ||
        ggml_cuda_is_btl3_avq2(unmarked)) {
        throw std::runtime_error("BTL3 CUDA marker contract mismatch");
    }
    ggml_tensor * int4_codes = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, 64, 128);
    ggml_tensor * int4_scales = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F16, 1, 128);
    ggml_tensor * int4_zeros = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, 1, 128);
    ggml_tensor * int4_output = btl3::build_int4_mul_mat(
        ctx.get(),
        int4_codes,
        int4_scales,
        int4_zeros,
        graph_input);
    ggml_tensor * vocab_codes = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, 32, 128);
    ggml_tensor * vocab_affine = ggml_new_tensor_3d(
        ctx.get(), GGML_TYPE_F32, 4, 4, 1);
    ggml_tensor * vocab_bias = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, 4, 1);
    ggml_tensor * vocab_input_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, 128);
    ggml_tensor * rescued_row_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * rescued_upper_six = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, 96, 1);
    ggml_tensor * rescued_scales = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_F16, 1);
    ggml_tensor * token_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, 2);
    ggml_tensor * vocab_rows = btl3::build_vocab_get_rows(
        ctx.get(),
        vocab_codes,
        vocab_affine,
        vocab_bias,
        vocab_input_signs,
        rescued_row_ids,
        rescued_upper_six,
        rescued_scales,
        token_ids);
    ggml_tensor * vocab_logits = btl3::build_vocab_head(
        ctx.get(),
        vocab_codes,
        vocab_affine,
        vocab_bias,
        vocab_input_signs,
        graph_input);
    if (!ggml_cuda_is_btl3_int4(int4_output) ||
        !ggml_cuda_is_btl3_vocab_rows(vocab_rows) ||
        !ggml_cuda_is_btl3_vocab_head(vocab_logits)) {
        throw std::runtime_error("BTL3 CUDA compact-op contract mismatch");
    }

    ggml_backend_load_all();
    ggml_backend_dev_t cpu_device =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(cpu_device, nullptr);
    if (backend == nullptr) {
        throw std::runtime_error("failed to create GGML CPU backend");
    }
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate GGML test tensors");
    }

    std::vector<int8_t> graph_code_data(128 * 32, static_cast<int8_t>(0xAA));
    std::array<float, 16> graph_affine_data = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    std::array<float, 4> graph_bias_data = {};
    std::vector<int8_t> graph_sign_data(128, 1);
    std::vector<float> graph_input_data(128 * 3);
    for (std::size_t token = 0; token < 3; ++token) {
        std::fill_n(
            graph_input_data.begin() + token * 128,
            128,
            static_cast<float>(token + 1));
    }

    ggml_backend_tensor_set(
        graph_codes, graph_code_data.data(), 0, graph_code_data.size());
    ggml_backend_tensor_set(
        graph_affine, graph_affine_data.data(), 0, sizeof(graph_affine_data));
    ggml_backend_tensor_set(
        graph_bias, graph_bias_data.data(), 0, sizeof(graph_bias_data));
    ggml_backend_tensor_set(
        graph_input_signs, graph_sign_data.data(), 0, graph_sign_data.size());
    ggml_backend_tensor_set(
        graph_output_signs, graph_sign_data.data(), 0, graph_sign_data.size());
    ggml_backend_tensor_set(
        graph_input,
        graph_input_data.data(),
        0,
        graph_input_data.size() * sizeof(float));
    std::vector<int8_t> int4_code_data(128 * 64, 0x11);
    std::vector<ggml_fp16_t> int4_scale_data(
        128, ggml_fp32_to_fp16(1.0F));
    std::vector<int8_t> int4_zero_data(128, 0);
    ggml_backend_tensor_set(
        int4_codes, int4_code_data.data(), 0, int4_code_data.size());
    ggml_backend_tensor_set(
        int4_scales,
        int4_scale_data.data(),
        0,
        int4_scale_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        int4_zeros, int4_zero_data.data(), 0, int4_zero_data.size());
    std::vector<int8_t> vocab_code_data(128 * 32, static_cast<int8_t>(0xAA));
    std::fill_n(
        vocab_code_data.begin() + 2 * 32,
        32,
        static_cast<int8_t>(0x55));
    std::vector<int8_t> upper_six_data(96);
    for (std::size_t group = 0; group < 32; ++group) {
        const uint32_t word =
            UINT32_C(32) |
            (UINT32_C(32) << 6) |
            (UINT32_C(32) << 12) |
            (UINT32_C(32) << 18);
        upper_six_data[group * 3] = static_cast<int8_t>(word & 0xFFU);
        upper_six_data[group * 3 + 1] =
            static_cast<int8_t>((word >> 8) & 0xFFU);
        upper_six_data[group * 3 + 2] =
            static_cast<int8_t>((word >> 16) & 0xFFU);
    }
    const int32_t rescue_id = 2;
    const ggml_fp16_t rescue_scale = ggml_fp32_to_fp16(1.0F);
    const std::array<int32_t, 2> vocab_token_ids = {1, 2};
    ggml_backend_tensor_set(
        vocab_codes, vocab_code_data.data(), 0, vocab_code_data.size());
    ggml_backend_tensor_set(
        vocab_affine, graph_affine_data.data(), 0, sizeof(graph_affine_data));
    ggml_backend_tensor_set(
        vocab_bias, graph_bias_data.data(), 0, sizeof(graph_bias_data));
    ggml_backend_tensor_set(
        vocab_input_signs,
        graph_sign_data.data(),
        0,
        graph_sign_data.size());
    ggml_backend_tensor_set(
        rescued_row_ids, &rescue_id, 0, sizeof(rescue_id));
    ggml_backend_tensor_set(
        rescued_upper_six,
        upper_six_data.data(),
        0,
        upper_six_data.size());
    ggml_backend_tensor_set(
        rescued_scales, &rescue_scale, 0, sizeof(rescue_scale));
    ggml_backend_tensor_set(
        token_ids, vocab_token_ids.data(), 0, sizeof(vocab_token_ids));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, graph_output);
    ggml_build_forward_expand(graph, int4_output);
    ggml_build_forward_expand(graph, vocab_rows);
    ggml_build_forward_expand(graph, vocab_logits);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        throw std::runtime_error("BTL3 AVQ2 GGML graph execution failed");
    }
    std::vector<float> graph_output_data(128 * 3);
    ggml_backend_tensor_get(
        graph_output,
        graph_output_data.data(),
        0,
        graph_output_data.size() * sizeof(float));
    for (std::size_t token = 0; token < 3; ++token) {
        require_close(
            graph_output_data[token * 128],
            64.0F * static_cast<float>(token + 1));
        for (std::size_t row = 1; row < 128; ++row) {
            require_close(graph_output_data[token * 128 + row], 0.0F);
        }
    }
    std::vector<float> int4_output_data(128 * 3);
    ggml_backend_tensor_get(
        int4_output,
        int4_output_data.data(),
        0,
        int4_output_data.size() * sizeof(float));
    for (std::size_t token = 0; token < 3; ++token) {
        const float expected_int4 = 128.0F * static_cast<float>(token + 1);
        for (std::size_t row = 0; row < 128; ++row) {
            require_close(
                int4_output_data[token * 128 + row],
                expected_int4);
        }
    }
    std::vector<float> vocab_row_data(128 * 2);
    ggml_backend_tensor_get(
        vocab_rows,
        vocab_row_data.data(),
        0,
        vocab_row_data.size() * sizeof(float));
    require_close(vocab_row_data[0], std::sqrt(128.0F) / 2.0F);
    for (std::size_t column = 1; column < 128; ++column) {
        require_close(vocab_row_data[column], 0.0F);
    }
    for (std::size_t column = 0; column < 128; ++column) {
        require_close(vocab_row_data[128 + column], 1.0F);
    }
    std::vector<float> vocab_logit_data(128 * 3);
    ggml_backend_tensor_get(
        vocab_logits,
        vocab_logit_data.data(),
        0,
        vocab_logit_data.size() * sizeof(float));
    for (std::size_t token = 0; token < 3; ++token) {
        require_close(
            vocab_logit_data[token * 128],
            std::sqrt(128.0F) / 2.0F *
                static_cast<float>(token + 1));
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}
