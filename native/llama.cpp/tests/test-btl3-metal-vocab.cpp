#include "../src/btl3-avq.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct fixture {
    int64_t vocabulary = 256;
    int64_t columns = 256;
    int64_t groups = 2;
    int64_t head_tokens = 2;
    std::vector<int8_t> codes;
    std::vector<float> affine;
    std::vector<float> bias;
    std::vector<int8_t> signs;
    std::vector<int32_t> rescued_ids;
    std::vector<int8_t> rescued_upper;
    std::vector<ggml_fp16_t> rescued_scales;
    std::vector<int32_t> token_ids;
    std::vector<float> head_input;
};

fixture make_fixture(
        int64_t vocabulary = 256,
        int64_t columns = 256,
        int64_t groups = 2,
        int64_t head_tokens = 2) {
    fixture value;
    value.vocabulary = vocabulary;
    value.columns = columns;
    value.groups = groups;
    value.head_tokens = head_tokens;
    value.codes.resize(value.vocabulary * value.columns / 4);
    value.affine.resize(value.groups * 16);
    value.bias.resize(value.groups * 4);
    value.signs.resize(value.columns);
    value.rescued_ids = {3, 200};
    value.rescued_upper.resize(
        value.rescued_ids.size() * value.columns * 3 / 4);
    value.rescued_scales = {
        ggml_fp32_to_fp16(0.004F),
        ggml_fp32_to_fp16(0.006F),
    };
    value.token_ids = {3, 100, 200};
    value.head_input.resize(value.columns * value.head_tokens);
    for (std::size_t i = 0; i < value.codes.size(); ++i) {
        value.codes[i] = static_cast<int8_t>((i * 37 + 11) & 0xFF);
    }
    for (std::size_t i = 0; i < value.affine.size(); ++i) {
        value.affine[i] = 0.01F * static_cast<float>(1 + i % 13);
    }
    for (std::size_t i = 0; i < value.bias.size(); ++i) {
        value.bias[i] = 0.002F * static_cast<float>(i) - 0.004F;
    }
    for (std::size_t i = 0; i < value.signs.size(); ++i) {
        value.signs[i] = i % 5 == 0 ? -1 : 1;
    }
    for (std::size_t i = 0; i < value.rescued_upper.size(); ++i) {
        value.rescued_upper[i] =
            static_cast<int8_t>((i * 17 + 23) & 0xFF);
    }
    for (std::size_t i = 0; i < value.head_input.size(); ++i) {
        value.head_input[i] =
            std::sin(static_cast<float>(i) * 0.021F);
    }
    return value;
}

std::vector<float> run(
        ggml_backend_dev_t device,
        const fixture & value,
        bool head,
        double * elapsed_ms = nullptr) {
    ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init(params), ggml_free);
    ggml_tensor * codes = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, value.columns / 4, value.vocabulary);
    ggml_tensor * affine = ggml_new_tensor_3d(
        ctx.get(), GGML_TYPE_F32, 4, 4, value.groups);
    ggml_tensor * bias = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, 4, value.groups);
    ggml_tensor * signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, value.columns);
    ggml_tensor * rescued_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, value.rescued_ids.size());
    ggml_tensor * rescued_upper = ggml_new_tensor_2d(
        ctx.get(),
        GGML_TYPE_I8,
        value.columns * 3 / 4,
        value.rescued_ids.size());
    ggml_tensor * rescued_scales = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_F16, value.rescued_scales.size());
    ggml_tensor * token_ids = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I32, value.token_ids.size());
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, value.columns, value.head_tokens);
    ggml_tensor * output = head
        ? btl3::build_vocab_head(
            ctx.get(), codes, affine, bias, signs, input)
        : btl3::build_vocab_get_rows(
            ctx.get(),
            codes,
            affine,
            bias,
            signs,
            rescued_ids,
            rescued_upper,
            rescued_scales,
            token_ids);
    if (!ggml_backend_dev_supports_op(device, output)) {
        throw std::runtime_error("backend rejected BTL3 vocabulary op");
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    ggml_backend_tensor_set(codes, value.codes.data(), 0, value.codes.size());
    ggml_backend_tensor_set(
        affine, value.affine.data(), 0, value.affine.size() * sizeof(float));
    ggml_backend_tensor_set(
        bias, value.bias.data(), 0, value.bias.size() * sizeof(float));
    ggml_backend_tensor_set(signs, value.signs.data(), 0, value.signs.size());
    ggml_backend_tensor_set(
        rescued_ids,
        value.rescued_ids.data(),
        0,
        value.rescued_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        rescued_upper,
        value.rescued_upper.data(),
        0,
        value.rescued_upper.size());
    ggml_backend_tensor_set(
        rescued_scales,
        value.rescued_scales.data(),
        0,
        value.rescued_scales.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        token_ids,
        value.token_ids.data(),
        0,
        value.token_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(
        input,
        value.head_input.data(),
        0,
        value.head_input.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("BTL3 vocabulary graph execution failed");
    }
    ggml_backend_synchronize(backend);
    if (elapsed_ms != nullptr) {
        const auto start = std::chrono::steady_clock::now();
        constexpr int iterations = 3;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (ggml_backend_graph_compute(backend, graph) !=
                GGML_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "BTL3 vocabulary benchmark execution failed");
            }
        }
        ggml_backend_synchronize(backend);
        const auto stop = std::chrono::steady_clock::now();
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            stop - start).count() / iterations;
    }
    std::vector<float> result(ggml_nelements(output));
    ggml_backend_tensor_get(
        output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return result;
}

void require_parity(
        const char * name,
        const std::vector<float> & actual,
        const std::vector<float> & expected) {
    float worst_absolute = 0.0F;
    float worst_relative = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float absolute = std::abs(actual[i] - expected[i]);
        const float relative =
            absolute / std::max(1.0F, std::abs(expected[i]));
        worst_absolute = std::max(worst_absolute, absolute);
        worst_relative = std::max(worst_relative, relative);
    }
    std::cout << name << " max_abs=" << worst_absolute
              << " max_rel=" << worst_relative << '\n';
    if (worst_relative > 2e-4F) {
        throw std::runtime_error("Metal vocabulary numerical parity failed");
    }
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t metal =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (cpu == nullptr || metal == nullptr) {
        throw std::runtime_error("required test backend is unavailable");
    }
    const bool benchmark =
        argc == 2 && std::string(argv[1]) == "--benchmark";
    if (benchmark) {
        const fixture value = make_fixture(248320, 5120, 194, 1);
        double elapsed_ms = 0.0;
        run(metal, value, true, &elapsed_ms);
        std::cout << "full_vocab_head metal_ms=" << elapsed_ms << '\n';
        return 0;
    }
    const fixture value = make_fixture();
    require_parity(
        "vocab_head",
        run(metal, value, true),
        run(cpu, value, true));
    require_parity(
        "vocab_rows",
        run(metal, value, false),
        run(cpu, value, false));
}
