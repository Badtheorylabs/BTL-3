#include "../src/btl3-avq-contract.h"
#include "../src/btl3-avq.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
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
            "CUDA INT4 mismatch: actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void custom_noop(ggml_tensor *, int, int, void *) {
}

} // namespace

int main() {
    constexpr std::size_t rows = 128;
    constexpr std::size_t columns = 256;
    constexpr std::size_t groups = columns / 128;
    constexpr std::size_t tokens = 3;
    std::vector<int8_t> codes(rows * columns / 2);
    std::vector<ggml_fp16_t> scales(rows * groups);
    std::vector<int8_t> zeros(rows * groups);
    std::vector<float> input(tokens * columns);
    for (std::size_t index = 0; index < codes.size(); ++index) {
        codes[index] = static_cast<int8_t>((index * 29 + 7) & 0xFFU);
    }
    for (std::size_t index = 0; index < scales.size(); ++index) {
        scales[index] = ggml_fp32_to_fp16(
            0.03125F * static_cast<float>(1 + index % 5));
        zeros[index] = static_cast<int8_t>(3 + index % 7);
    }
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::cos(static_cast<float>(index) * 0.013F);
    }

    std::vector<float> expected(rows * tokens);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t row = 0; row < rows; ++row) {
            float sum = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                const uint8_t packed = static_cast<uint8_t>(
                    codes[row * (columns / 2) + column / 2]);
                const uint8_t code = (column & 1) == 0
                    ? packed & 0x0FU
                    : packed >> 4;
                const std::size_t parameter =
                    row * groups + column / 128;
                const float weight =
                    (static_cast<float>(code) -
                     static_cast<float>(
                         static_cast<uint8_t>(zeros[parameter]))) *
                    ggml_fp16_to_fp32(scales[parameter]);
                sum += weight * input[token * columns + column];
            }
            expected[token * rows + row] = sum;
        }
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
        ctx.get(), GGML_TYPE_I8, columns / 2, rows);
    ggml_tensor * graph_scales = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F16, groups, rows);
    ggml_tensor * graph_zeros = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, groups, rows);
    ggml_tensor * graph_input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, columns, tokens);
    ggml_tensor * output = btl3::build_int4_mul_mat(
        ctx.get(), graph_codes, graph_scales, graph_zeros, graph_input);
    ggml_tensor * probe_args[] = {graph_codes};
    ggml_tensor * probe = ggml_custom_4d(
        ctx.get(), GGML_TYPE_F32, 1, 1, 1, 1,
        probe_args, 1, custom_noop, 1,
        reinterpret_cast<void *>(btl3::cuda_int4_probe_custom_magic));
    if (!ggml_backend_dev_supports_op(device, probe) ||
        !ggml_backend_dev_supports_op(device, output)) {
        ggml_backend_free(backend);
        throw std::runtime_error("CUDA backend rejected BTL3 INT4");
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate CUDA test tensors");
    }
    ggml_backend_tensor_set(graph_codes, codes.data(), 0, codes.size());
    ggml_backend_tensor_set(
        graph_scales, scales.data(), 0,
        scales.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(graph_zeros, zeros.data(), 0, zeros.size());
    ggml_backend_tensor_set(
        graph_input, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        throw std::runtime_error("CUDA INT4 graph execution failed");
    }

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(
        output, actual.data(), 0, actual.size() * sizeof(float));
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require_close(actual[index], expected[index]);
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}
