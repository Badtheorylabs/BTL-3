#include "../src/btl3-avq-contract.h"
#include "../src/btl3-avq.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(float actual, float expected) {
    const float tolerance = 2.0e-3F * std::max(1.0F, std::abs(expected));
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "CUDA AVQ2 mismatch: actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void custom_noop(ggml_tensor *, int, int, void *) {
}

} // namespace

int main() {
    constexpr std::size_t rows = 256;
    constexpr std::size_t columns = 256;
    constexpr std::size_t codebook_rows = 128;
    constexpr std::size_t groups = rows / codebook_rows;
    constexpr std::size_t tokens = 3;
    constexpr std::size_t codes_per_row = columns / 4;

    std::vector<int8_t> codes(rows * codes_per_row);
    for (std::size_t index = 0; index < codes.size(); ++index) {
        codes[index] =
            static_cast<int8_t>((index * 73 + 19) & 0xFFU);
    }
    std::vector<float> affine_weight(groups * 16);
    std::vector<float> affine_bias(groups * 4);
    for (std::size_t index = 0; index < affine_weight.size(); ++index) {
        affine_weight[index] =
            (static_cast<float>(index % 11) - 5.0F) * 0.03125F;
    }
    for (std::size_t index = 0; index < affine_bias.size(); ++index) {
        affine_bias[index] =
            (static_cast<float>(index % 5) - 2.0F) * 0.0625F;
    }
    std::vector<int8_t> input_signs(columns);
    std::vector<int8_t> output_signs(rows);
    for (std::size_t index = 0; index < columns; ++index) {
        input_signs[index] = index % 3 == 0 ? -1 : 1;
    }
    for (std::size_t index = 0; index < rows; ++index) {
        output_signs[index] = index % 5 == 0 ? -1 : 1;
    }
    std::vector<float> input(tokens * columns);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            std::sin(static_cast<float>(index) * 0.017F) * 0.75F;
    }

    const btl3::avq2_matrix cpu_matrix = {
        reinterpret_cast<const uint8_t *>(codes.data()),
        affine_weight.data(),
        affine_bias.data(),
        input_signs.data(),
        output_signs.data(),
        rows,
        columns,
        codebook_rows,
        btl3::avq2_transform_block,
    };
    std::vector<float> expected(tokens * rows);
    for (std::size_t token = 0; token < tokens; ++token) {
        btl3::mul_mat_vec(
            cpu_matrix,
            input.data() + token * columns,
            expected.data() + token * rows);
    }

    ggml_backend_load_all();
    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    if (cuda_reg == nullptr || ggml_backend_reg_dev_count(cuda_reg) == 0) {
        throw std::runtime_error("CUDA backend is unavailable");
    }
    ggml_backend_dev_t cuda_device = ggml_backend_reg_dev_get(cuda_reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(cuda_device, nullptr);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize CUDA backend");
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
        ggml_backend_free(backend);
        throw std::runtime_error("failed to create GGML test context");
    }

    ggml_tensor * graph_codes = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, codes_per_row, rows);
    ggml_tensor * graph_affine = ggml_new_tensor_3d(
        ctx.get(), GGML_TYPE_F32, 4, 4, groups);
    ggml_tensor * graph_bias = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, 4, groups);
    ggml_tensor * graph_input_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, columns);
    ggml_tensor * graph_output_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, rows);
    ggml_tensor * graph_input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, columns, tokens);
    ggml_tensor * graph_output = btl3::build_mul_mat(
        ctx.get(),
        graph_codes,
        graph_affine,
        graph_bias,
        graph_input_signs,
        graph_output_signs,
        graph_input);

    ggml_tensor * probe_args[] = {graph_codes};
    ggml_tensor * probe = ggml_custom_4d(
        ctx.get(),
        GGML_TYPE_F32,
        1,
        1,
        1,
        1,
        probe_args,
        1,
        custom_noop,
        1,
        reinterpret_cast<void *>(btl3::cuda_avq2_probe_custom_magic));
    if (!ggml_backend_dev_supports_op(cuda_device, probe) ||
        !ggml_backend_dev_supports_op(cuda_device, graph_output)) {
        ggml_backend_free(backend);
        throw std::runtime_error("CUDA backend rejected BTL3 AVQ2");
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate CUDA test tensors");
    }
    ggml_backend_tensor_set(
        graph_codes, codes.data(), 0, codes.size());
    ggml_backend_tensor_set(
        graph_affine,
        affine_weight.data(),
        0,
        affine_weight.size() * sizeof(float));
    ggml_backend_tensor_set(
        graph_bias,
        affine_bias.data(),
        0,
        affine_bias.size() * sizeof(float));
    ggml_backend_tensor_set(
        graph_input_signs,
        input_signs.data(),
        0,
        input_signs.size());
    ggml_backend_tensor_set(
        graph_output_signs,
        output_signs.data(),
        0,
        output_signs.size());
    ggml_backend_tensor_set(
        graph_input,
        input.data(),
        0,
        input.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, graph_output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        throw std::runtime_error("CUDA AVQ2 graph execution failed");
    }

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(
        graph_output,
        actual.data(),
        0,
        actual.size() * sizeof(float));
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require_close(actual[index], expected[index]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}
