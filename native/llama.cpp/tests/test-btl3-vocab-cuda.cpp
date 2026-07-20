#include "../src/btl3-avq-contract.h"
#include "../src/btl3-avq.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(float actual, float expected) {
    const float tolerance = 2.0e-3F * std::max(1.0F, std::abs(expected));
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "CUDA vocabulary mismatch: actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void custom_noop(ggml_tensor *, int, int, void *) {
}

} // namespace

int main() {
    constexpr std::size_t columns = 128;
    constexpr std::size_t vocabulary = 128;
    constexpr std::size_t tokens = 3;
    std::vector<int8_t> codes(vocabulary * columns / 4, int8_t(0xAA));
    std::fill_n(codes.begin() + 2 * columns / 4, columns / 4, int8_t(0x55));
    const std::array<float, 16> affine = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    const std::array<float, 4> bias = {};
    std::vector<int8_t> signs(columns, 1);
    signs[0] = -1;
    std::vector<int8_t> upper(columns * 3 / 4);
    for (std::size_t group = 0; group < columns / 4; ++group) {
        const uint32_t word =
            UINT32_C(32) | (UINT32_C(32) << 6) |
            (UINT32_C(32) << 12) | (UINT32_C(32) << 18);
        upper[group * 3] = static_cast<int8_t>(word);
        upper[group * 3 + 1] = static_cast<int8_t>(word >> 8);
        upper[group * 3 + 2] = static_cast<int8_t>(word >> 16);
    }
    const int32_t rescue_id = 2;
    const ggml_fp16_t rescue_scale = ggml_fp32_to_fp16(1.0F);
    const std::array<int32_t, tokens> token_ids = {1, 2, -1};
    std::vector<float> input(tokens * columns);
    for (std::size_t token = 0; token < tokens; ++token) {
        std::fill_n(
            input.begin() + token * columns,
            columns,
            static_cast<float>(token + 1));
    }

    ggml_backend_load_all();
    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    if (cuda_reg == nullptr || ggml_backend_reg_dev_count(cuda_reg) == 0) {
        throw std::runtime_error("CUDA backend is unavailable");
    }
    ggml_backend_dev_t device = ggml_backend_reg_dev_get(cuda_reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize CUDA backend");
    }
    ggml_init_params params = {1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init(params), ggml_free);

    ggml_tensor * graph_codes = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, columns / 4, vocabulary);
    ggml_tensor * graph_affine = ggml_new_tensor_3d(
        ctx.get(), GGML_TYPE_F32, 4, 4, 1);
    ggml_tensor * graph_bias = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, 4, 1);
    ggml_tensor * graph_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, columns);
    ggml_tensor * graph_rescued_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * graph_upper = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, columns * 3 / 4, 1);
    ggml_tensor * graph_rescued_scales = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_F16, 1);
    ggml_tensor * graph_token_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * graph_input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, columns, tokens);
    ggml_tensor * rows = btl3::build_vocab_get_rows(
        ctx.get(), graph_codes, graph_affine, graph_bias, graph_signs,
        graph_rescued_ids, graph_upper, graph_rescued_scales,
        graph_token_ids);
    ggml_tensor * logits = btl3::build_vocab_head(
        ctx.get(), graph_codes, graph_affine, graph_bias, graph_signs,
        graph_input);

    auto make_probe = [&](std::uintptr_t magic) {
        ggml_tensor * args[] = {graph_codes};
        return ggml_custom_4d(
            ctx.get(), GGML_TYPE_F32, 1, 1, 1, 1,
            args, 1, custom_noop, 1, reinterpret_cast<void *>(magic));
    };
    ggml_tensor * rows_probe = make_probe(
        btl3::cuda_vocab_rows_probe_custom_magic);
    ggml_tensor * head_probe = make_probe(
        btl3::cuda_vocab_head_probe_custom_magic);
    if (!ggml_backend_dev_supports_op(device, rows_probe) ||
        !ggml_backend_dev_supports_op(device, head_probe) ||
        !ggml_backend_dev_supports_op(device, rows) ||
        !ggml_backend_dev_supports_op(device, logits)) {
        ggml_backend_free(backend);
        throw std::runtime_error("CUDA backend rejected BTL3 vocabulary ops");
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate CUDA test tensors");
    }
    ggml_backend_tensor_set(graph_codes, codes.data(), 0, codes.size());
    ggml_backend_tensor_set(
        graph_affine, affine.data(), 0, sizeof(affine));
    ggml_backend_tensor_set(graph_bias, bias.data(), 0, sizeof(bias));
    ggml_backend_tensor_set(graph_signs, signs.data(), 0, signs.size());
    ggml_backend_tensor_set(
        graph_rescued_ids, &rescue_id, 0, sizeof(rescue_id));
    ggml_backend_tensor_set(graph_upper, upper.data(), 0, upper.size());
    ggml_backend_tensor_set(
        graph_rescued_scales, &rescue_scale, 0, sizeof(rescue_scale));
    ggml_backend_tensor_set(
        graph_token_ids, token_ids.data(), 0, sizeof(token_ids));
    ggml_backend_tensor_set(
        graph_input, input.data(), 0, input.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, rows);
    ggml_build_forward_expand(graph, logits);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        throw std::runtime_error("CUDA vocabulary graph execution failed");
    }

    std::vector<float> row_output(tokens * columns);
    ggml_backend_tensor_get(
        rows, row_output.data(), 0, row_output.size() * sizeof(float));
    require_close(row_output[0], -std::sqrt(128.0F) / 2.0F);
    for (std::size_t column = 1; column < columns; ++column) {
        require_close(row_output[column], 0.0F);
    }
    for (std::size_t column = 0; column < columns; ++column) {
        require_close(row_output[columns + column], 1.0F);
        require_close(row_output[2 * columns + column], 0.0F);
    }

    std::vector<float> logit_output(tokens * vocabulary);
    ggml_backend_tensor_get(
        logits, logit_output.data(), 0,
        logit_output.size() * sizeof(float));
    for (std::size_t token = 0; token < tokens; ++token) {
        const float scale = static_cast<float>(token + 1);
        require_close(
            logit_output[token * vocabulary],
            -std::sqrt(128.0F) / 2.0F * scale);
        require_close(
            logit_output[token * vocabulary + 2],
            std::sqrt(128.0F) / 2.0F * scale);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}
