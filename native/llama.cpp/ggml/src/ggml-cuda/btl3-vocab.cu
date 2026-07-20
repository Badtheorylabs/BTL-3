#include "btl3-vocab.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

constexpr int transform_block = 128;
constexpr float transform_scale = 0.08838834764831845F;

__device__ void fwht(float * values, int lane) {
    for (int width = 1; width < transform_block; width *= 2) {
        const int pair = lane / width;
        if ((pair & 1) == 0) {
            const int right = lane + width;
            const float left_value = values[lane];
            const float right_value = values[right];
            values[lane] = left_value + right_value;
            values[right] = left_value - right_value;
        }
        __syncthreads();
    }
}

__device__ float decode_component(
        uint8_t code,
        int component,
        const float * affine,
        const float * bias) {
    float value = bias[component];
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const int shift = 6 - index * 2;
        const float lattice =
            static_cast<float>((code >> shift) & 3U) - 1.5F;
        value = fmaf(
            lattice,
            affine[component * 4 + index],
            value);
    }
    return value;
}

__device__ int find_rescue(
        const int32_t * row_ids,
        int64_t count,
        int32_t token) {
    int64_t left = 0;
    int64_t right = count;
    while (left < right) {
        const int64_t middle = left + (right - left) / 2;
        if (row_ids[middle] < token) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left < count && row_ids[left] == token
        ? static_cast<int>(left)
        : -1;
}

__global__ void vocab_get_rows(
        const uint8_t * codes,
        const float * affine,
        const float * bias,
        const int8_t * signs,
        const int32_t * rescued_ids,
        const uint8_t * rescued_upper,
        const half * rescued_scales,
        const int32_t * token_ids,
        float * output,
        int64_t columns,
        int64_t vocabulary,
        int64_t groups,
        int64_t rescue_count,
        int64_t tokens) {
    __shared__ float values[transform_block];
    __shared__ float group_affine[16];
    __shared__ float group_bias[4];
    __shared__ int rescue_index;
    const int lane = threadIdx.x;
    const int64_t blocks = columns / transform_block;
    const int64_t token_index = blockIdx.x / blocks;
    const int64_t column_block = blockIdx.x % blocks;
    if (token_index >= tokens) {
        return;
    }
    const int32_t token = token_ids[token_index];
    const int64_t column = column_block * transform_block + lane;
    float * row_output = output + token_index * columns;

    if (lane == 0) {
        rescue_index = token >= 0 && token < vocabulary
            ? find_rescue(rescued_ids, rescue_count, token)
            : -2;
    }
    __syncthreads();
    if (rescue_index == -2) {
        row_output[column] = 0.0F;
        return;
    }

    if (rescue_index >= 0) {
        const int64_t code_offset =
            static_cast<int64_t>(token) * (columns / 4) + column / 4;
        const uint8_t low =
            (codes[code_offset] >> ((column & 3) * 2)) & 3U;
        const int64_t upper_stride = columns * 3 / 4;
        const int64_t upper_group = column / 4;
        const uint8_t * packed =
            rescued_upper + rescue_index * upper_stride + upper_group * 3;
        const uint32_t word =
            static_cast<uint32_t>(packed[0]) |
            (static_cast<uint32_t>(packed[1]) << 8) |
            (static_cast<uint32_t>(packed[2]) << 16);
        const uint8_t upper =
            static_cast<uint8_t>((word >> ((column & 3) * 6)) & 0x3FU);
        const uint8_t value = low | static_cast<uint8_t>(upper << 2);
        row_output[column] =
            (static_cast<float>(value) - 128.0F) *
            __half2float(rescued_scales[rescue_index]);
        return;
    }

    const int64_t codebook_rows = vocabulary / groups;
    const int64_t group = token / codebook_rows;
    if (lane < 16) {
        group_affine[lane] = affine[group * 16 + lane];
    }
    if (lane < 4) {
        group_bias[lane] = bias[group * 4 + lane];
    }
    __syncthreads();
    const uint8_t code =
        codes[static_cast<int64_t>(token) * (columns / 4) + column / 4];
    values[lane] = decode_component(
        code,
        column & 3,
        group_affine,
        group_bias);
    __syncthreads();
    fwht(values, lane);
    row_output[column] =
        values[lane] * transform_scale * static_cast<float>(signs[column]);
}

__global__ void vocab_head(
        const uint8_t * codes,
        const float * affine,
        const float * bias,
        const int8_t * signs,
        const float * input,
        float * output,
        int64_t columns,
        int64_t vocabulary,
        int64_t groups,
        int64_t tokens) {
    __shared__ float values[transform_block];
    __shared__ float group_affine[16];
    __shared__ float group_bias[4];
    const int lane = threadIdx.x;
    const int64_t row_blocks = vocabulary / transform_block;
    const int64_t token = blockIdx.x / row_blocks;
    const int64_t row =
        (blockIdx.x % row_blocks) * transform_block + lane;
    if (token >= tokens) {
        return;
    }
    const int64_t codebook_rows = vocabulary / groups;
    const int64_t group = row / codebook_rows;
    if (lane < 16) {
        group_affine[lane] = affine[group * 16 + lane];
    }
    if (lane < 4) {
        group_bias[lane] = bias[group * 4 + lane];
    }
    __syncthreads();

    float accumulator = 0.0F;
    for (int64_t block = 0;
            block < columns / transform_block;
            ++block) {
        const int64_t block_start = block * transform_block;
        const int64_t column = block_start + lane;
        values[lane] =
            input[token * columns + column] *
            static_cast<float>(signs[column]);
        __syncthreads();
        fwht(values, lane);
        const int64_t code_start =
            row * (columns / 4) + block_start / 4;
        for (int code_index = 0;
                code_index < transform_block / 4;
                ++code_index) {
            const uint8_t code = codes[code_start + code_index];
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                accumulator = fmaf(
                    decode_component(
                        code,
                        component,
                        group_affine,
                        group_bias),
                    values[code_index * 4 + component] * transform_scale,
                    accumulator);
            }
        }
        __syncthreads();
    }
    output[token * vocabulary + row] = accumulator;
}

} // namespace

void ggml_cuda_op_btl3_vocab_rows(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_is_btl3_vocab_rows(dst));
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * affine = dst->src[1];
    const ggml_tensor * bias = dst->src[2];
    const ggml_tensor * signs = dst->src[3];
    const ggml_tensor * rescued_ids = dst->src[4];
    const ggml_tensor * rescued_upper = dst->src[5];
    const ggml_tensor * rescued_scales = dst->src[6];
    const ggml_tensor * token_ids = dst->src[7];
    const int64_t columns = dst->ne[0];
    const int64_t vocabulary = codes->ne[1];
    const int64_t groups = bias->ne[1];
    const int64_t tokens = dst->ne[1] * dst->ne[2] * dst->ne[3];

    vocab_get_rows<<<tokens * (columns / transform_block),
            transform_block, 0, ctx.stream()>>>(
        static_cast<const uint8_t *>(codes->data),
        static_cast<const float *>(affine->data),
        static_cast<const float *>(bias->data),
        static_cast<const int8_t *>(signs->data),
        static_cast<const int32_t *>(rescued_ids->data),
        static_cast<const uint8_t *>(rescued_upper->data),
        static_cast<const half *>(rescued_scales->data),
        static_cast<const int32_t *>(token_ids->data),
        static_cast<float *>(dst->data),
        columns,
        vocabulary,
        groups,
        rescued_ids->ne[0],
        tokens);
}

void ggml_cuda_op_btl3_vocab_head(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_is_btl3_vocab_head(dst));
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * affine = dst->src[1];
    const ggml_tensor * bias = dst->src[2];
    const ggml_tensor * signs = dst->src[3];
    const ggml_tensor * input = dst->src[4];
    const int64_t columns = input->ne[0];
    const int64_t vocabulary = dst->ne[0];
    const int64_t groups = bias->ne[1];
    const int64_t tokens = input->ne[1] * input->ne[2] * input->ne[3];

    vocab_head<<<tokens * (vocabulary / transform_block),
            transform_block, 0, ctx.stream()>>>(
        static_cast<const uint8_t *>(codes->data),
        static_cast<const float *>(affine->data),
        static_cast<const float *>(bias->data),
        static_cast<const int8_t *>(signs->data),
        static_cast<const float *>(input->data),
        static_cast<float *>(dst->data),
        columns,
        vocabulary,
        groups,
        tokens);
}

#else

void ggml_cuda_op_btl3_vocab_rows(
        ggml_backend_cuda_context &,
        ggml_tensor *) {
    GGML_ABORT("BTL3 vocabulary get-rows is only implemented for CUDA");
}

void ggml_cuda_op_btl3_vocab_head(
        ggml_backend_cuda_context &,
        ggml_tensor *) {
    GGML_ABORT("BTL3 vocabulary head is only implemented for CUDA");
}

#endif
