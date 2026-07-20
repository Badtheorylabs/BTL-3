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

constexpr std::uintptr_t packed_probe_magic =
    UINT64_C(0x42544c3350524f42);
constexpr std::uintptr_t metal_packed_probe_magic =
    UINT64_C(0x42544c334d505242);

void packed_probe_noop(ggml_tensor *, int, int, void *) {
}

void require_packed_probe_support(
        ggml_backend_dev_t device,
        std::uintptr_t magic) {
    ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init(params), ggml_free);
    ggml_tensor * packed =
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, 64, 64);
    ggml_tensor * sources[] = {packed};
    ggml_tensor * probe = ggml_custom_4d(
        ctx.get(),
        GGML_TYPE_F32,
        1,
        1,
        1,
        1,
        sources,
        1,
        packed_probe_noop,
        1,
        reinterpret_cast<void *>(magic));
    if (!ggml_backend_dev_supports_op(device, probe)) {
        throw std::runtime_error(
            "Metal backend rejected the packed-weight placement probe");
    }
}

struct fixture {
    int64_t rows;
    int64_t columns;
    int64_t groups;
    int64_t tokens;
    std::vector<int8_t> codes;
    std::vector<float> affine;
    std::vector<float> bias;
    std::vector<int8_t> input_signs;
    std::vector<int8_t> output_signs;
    std::vector<float> input;
};

fixture make_fixture(
        int64_t rows,
        int64_t columns,
        int64_t tokens,
        int64_t requested_groups = 0) {
    const int64_t groups =
        requested_groups > 0 ? requested_groups : rows / 128;
    fixture value = {
        rows,
        columns,
        groups,
        tokens,
        std::vector<int8_t>(rows * columns / 4),
        std::vector<float>(groups * 16),
        std::vector<float>(groups * 4),
        std::vector<int8_t>(columns),
        std::vector<int8_t>(rows),
        std::vector<float>(columns * tokens),
    };
    for (std::size_t i = 0; i < value.codes.size(); ++i) {
        value.codes[i] = static_cast<int8_t>((i * 73 + 19) & 0xFF);
    }
    for (int64_t group = 0; group < groups; ++group) {
        for (int64_t component = 0; component < 4; ++component) {
            value.bias[group * 4 + component] =
                0.003F * static_cast<float>(component - group);
            for (int64_t lattice = 0; lattice < 4; ++lattice) {
                value.affine[group * 16 + component * 4 + lattice] =
                    0.0125F * static_cast<float>(
                        1 + component + lattice + group);
            }
        }
    }
    for (int64_t i = 0; i < columns; ++i) {
        value.input_signs[i] = i % 3 == 0 ? -1 : 1;
    }
    for (int64_t i = 0; i < rows; ++i) {
        value.output_signs[i] = i % 5 == 0 ? -1 : 1;
    }
    for (int64_t i = 0; i < columns * tokens; ++i) {
        value.input[i] = std::sin(static_cast<float>(i) * 0.013F);
    }
    return value;
}

std::vector<float> reference(const fixture & value) {
    const btl3::avq2_matrix matrix = {
        reinterpret_cast<const uint8_t *>(value.codes.data()),
        value.affine.data(),
        value.bias.data(),
        value.input_signs.data(),
        value.output_signs.data(),
        static_cast<std::size_t>(value.rows),
        static_cast<std::size_t>(value.columns),
        static_cast<std::size_t>(value.rows / value.groups),
        128,
    };
    std::vector<float> output(value.rows * value.tokens);
    for (int64_t token = 0; token < value.tokens; ++token) {
        btl3::mul_mat_vec(
            matrix,
            value.input.data() + token * value.columns,
            output.data() + token * value.rows);
    }
    return output;
}

std::vector<float> run_metal(
        ggml_backend_dev_t device,
        const fixture & value,
        int iterations,
        double & elapsed_ms) {
    const std::size_t arena_bytes = 4 * 1024 * 1024;
    ggml_init_params params = {arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init(params), ggml_free);
    ggml_tensor * codes = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, value.columns / 4, value.rows);
    ggml_tensor * affine = ggml_new_tensor_3d(
        ctx.get(), GGML_TYPE_F32, 4, 4, value.groups);
    ggml_tensor * bias = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, 4, value.groups);
    ggml_tensor * input_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, value.columns);
    ggml_tensor * output_signs = ggml_new_tensor_1d(
        ctx.get(), GGML_TYPE_I8, value.rows);
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, value.columns, value.tokens);
    ggml_tensor * output = btl3::build_mul_mat(
        ctx.get(),
        codes,
        affine,
        bias,
        input_signs,
        output_signs,
        input);
    if (!ggml_backend_dev_supports_op(device, output)) {
        throw std::runtime_error("Metal backend rejected BTL3 AVQ2");
    }

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    ggml_backend_tensor_set(
        codes, value.codes.data(), 0, value.codes.size());
    ggml_backend_tensor_set(
        affine, value.affine.data(), 0, value.affine.size() * sizeof(float));
    ggml_backend_tensor_set(
        bias, value.bias.data(), 0, value.bias.size() * sizeof(float));
    ggml_backend_tensor_set(
        input_signs, value.input_signs.data(), 0, value.input_signs.size());
    ggml_backend_tensor_set(
        output_signs, value.output_signs.data(), 0, value.output_signs.size());
    ggml_backend_tensor_set(
        input, value.input.data(), 0, value.input.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Metal AVQ2 warmup failed");
    }
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Metal AVQ2 graph execution failed");
        }
        ggml_backend_synchronize(backend);
    }
    const auto stop = std::chrono::steady_clock::now();
    elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count() /
        iterations;

    std::vector<float> result(value.rows * value.tokens);
    ggml_backend_tensor_get(
        output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return result;
}

void require_parity(
        const std::vector<float> & actual,
        const std::vector<float> & expected) {
    float worst_absolute = 0.0F;
    float worst_relative = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float absolute = std::abs(actual[i] - expected[i]);
        const float relative = absolute / std::max(1.0F, std::abs(expected[i]));
        worst_absolute = std::max(worst_absolute, absolute);
        worst_relative = std::max(worst_relative, relative);
    }
    std::cout << "max_abs=" << worst_absolute
              << " max_rel=" << worst_relative << '\n';
    if (worst_relative > 2e-4F) {
        throw std::runtime_error("Metal AVQ2 numerical parity failed");
    }
}

double run_int4(
        ggml_backend_dev_t device,
        int64_t rows,
        int64_t columns,
        int64_t tokens,
        int iterations,
        bool check_parity) {
    const int64_t groups = columns / 128;
    std::vector<int8_t> codes(rows * columns / 2);
    std::vector<ggml_fp16_t> scales(rows * groups);
    std::vector<int8_t> zeros(rows * groups);
    std::vector<float> input(columns * tokens);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<int8_t>((i * 29 + 7) & 0xFF);
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = ggml_fp32_to_fp16(
            0.0025F * static_cast<float>(1 + i % 11));
        zeros[i] = static_cast<int8_t>(5 + i % 6);
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = std::cos(static_cast<float>(i) * 0.017F);
    }
    std::vector<float> expected;
    if (check_parity) {
        expected.assign(rows * tokens, 0.0F);
        for (int64_t token = 0; token < tokens; ++token) {
          for (int64_t row = 0; row < rows; ++row) {
            for (int64_t column = 0; column < columns; ++column) {
                const uint8_t packed = static_cast<uint8_t>(
                    codes[row * columns / 2 + column / 2]);
                const uint8_t code =
                    column % 2 == 0 ? packed & 15U : packed >> 4;
                const int64_t group = row * groups + column / 128;
                expected[token * rows + row] +=
                    (static_cast<float>(code) -
                     static_cast<float>(
                         static_cast<uint8_t>(zeros[group]))) *
                    ggml_fp16_to_fp32(scales[group]) *
                    input[token * columns + column];
            }
          }
        }
    }

    ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init(params), ggml_free);
    ggml_tensor * code_tensor = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, columns / 2, rows);
    ggml_tensor * scale_tensor = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F16, groups, rows);
    ggml_tensor * zero_tensor = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I8, groups, rows);
    ggml_tensor * input_tensor = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, columns, tokens);
    ggml_tensor * output = btl3::build_int4_mul_mat(
        ctx.get(),
        code_tensor,
        scale_tensor,
        zero_tensor,
        input_tensor);
    if (!ggml_backend_dev_supports_op(device, output)) {
        throw std::runtime_error("Metal backend rejected BTL3 INT4");
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    ggml_backend_tensor_set(
        code_tensor, codes.data(), 0, codes.size());
    ggml_backend_tensor_set(
        scale_tensor, scales.data(), 0,
        scales.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
        zero_tensor, zeros.data(), 0, zeros.size());
    ggml_backend_tensor_set(
        input_tensor, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Metal INT4 graph execution failed");
    }
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (ggml_backend_graph_compute(backend, graph) !=
            GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Metal INT4 benchmark failed");
        }
    }
    ggml_backend_synchronize(backend);
    const auto stop = std::chrono::steady_clock::now();
    if (check_parity) {
        std::vector<float> actual(rows * tokens);
        ggml_backend_tensor_get(
            output, actual.data(), 0, actual.size() * sizeof(float));
        require_parity(actual, expected);
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return std::chrono::duration<double, std::milli>(
        stop - start).count() / iterations;
}

} // namespace

int main(int argc, char ** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
    const bool suite = argc == 2 && std::string(argv[1]) == "--suite";
    ggml_backend_load_all();
    ggml_backend_dev_t device =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr) {
        throw std::runtime_error("Metal GPU device is unavailable");
    }
    require_packed_probe_support(device, packed_probe_magic);
    require_packed_probe_support(device, metal_packed_probe_magic);
    run_int4(device, 256, 256, 3, 1, true);
    if (suite) {
        struct shape {
            int64_t rows;
            int64_t columns;
            int64_t groups;
            int count;
        };
        const shape shapes[] = {
            {5120, 6144, 5, 45},
            {5120, 17408, 5, 62},
            {6144, 5120, 6, 44},
            {10240, 5120, 10, 46},
            {17408, 5120, 17, 126},
        };
        double weighted_ms = 0.0;
        for (const shape & item : shapes) {
            const fixture sample = make_fixture(
                item.rows, item.columns, 1, item.groups);
            double elapsed_ms = 0.0;
            run_metal(device, sample, 3, elapsed_ms);
            weighted_ms += elapsed_ms * item.count;
            std::cout << "shape=" << item.rows << 'x' << item.columns
                      << " count=" << item.count
                      << " metal_ms=" << elapsed_ms << '\n';
        }
        std::cout << "weighted_avq2_decoder_ms=" << weighted_ms << '\n';
        const shape int4_shapes[] = {
            {1024, 5120, 0, 31},
            {5120, 6144, 0, 15},
            {6144, 5120, 0, 2},
            {10240, 5120, 0, 1},
            {12288, 5120, 0, 16},
        };
        double weighted_int4_ms = 0.0;
        for (const shape & item : int4_shapes) {
            const double elapsed_ms =
                run_int4(device, item.rows, item.columns, 1, 3, false);
            weighted_int4_ms += elapsed_ms * item.count;
            std::cout << "int4_shape=" << item.rows << 'x' << item.columns
                      << " count=" << item.count
                      << " metal_ms=" << elapsed_ms << '\n';
        }
        std::cout << "weighted_int4_decoder_ms="
                  << weighted_int4_ms << '\n';
        return 0;
    }
    const fixture value = benchmark
        ? make_fixture(4096, 4096, 4)
        : make_fixture(256, 256, 5);
    const auto cpu_start = std::chrono::steady_clock::now();
    const std::vector<float> expected = reference(value);
    const auto cpu_stop = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(
            cpu_stop - cpu_start).count();
    double elapsed_ms = 0.0;
    const std::vector<float> actual =
        run_metal(device, value, benchmark ? 20 : 3, elapsed_ms);
    require_parity(actual, expected);
    std::cout << "device=" << ggml_backend_dev_name(device)
              << " rows=" << value.rows
              << " columns=" << value.columns
              << " tokens=" << value.tokens
              << " cpu_ms=" << cpu_ms
              << " metal_ms=" << elapsed_ms
              << " speedup=" << cpu_ms / elapsed_ms << '\n';
}
