#include "btl3-avq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

constexpr int transform_block = 128;
constexpr int matvec_threads = 256;
constexpr float transform_scale = 0.08838834764831845F;

__global__ void transform_input(
        const float * input,
        const int8_t * signs,
        float * transformed,
        int64_t columns,
        int64_t blocks_per_token) {
    __shared__ float values[transform_block];
    const int64_t token = blockIdx.x / blocks_per_token;
    const int64_t token_block = blockIdx.x % blocks_per_token;
    const int64_t block_start = token_block * transform_block;
    const int lane = threadIdx.x;
    const int64_t index = token * columns + block_start + lane;
    values[lane] =
        input[index] * static_cast<float>(signs[block_start + lane]);
    __syncthreads();

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
    transformed[index] = values[lane] * transform_scale;
}

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

__global__ void decode_mul_mat(
        const uint8_t * codes,
        const float * affine_weight,
        const float * affine_bias,
        const float * input,
        float * output,
        int64_t rows,
        int64_t columns,
        int64_t codebook_rows) {
    __shared__ float warp_sums[matvec_threads / 32];
    __shared__ float group_weight[16];
    __shared__ float group_bias[4];
    const int64_t row = blockIdx.x % rows;
    const int64_t token = blockIdx.x / rows;
    const int64_t group = row / codebook_rows;
    const int64_t codes_per_row = columns / 4;
    if (threadIdx.x < 16) {
        group_weight[threadIdx.x] =
            affine_weight[group * 16 + threadIdx.x];
    }
    if (threadIdx.x < 4) {
        group_bias[threadIdx.x] =
            affine_bias[group * 4 + threadIdx.x];
    }
    __syncthreads();

    const float * token_input = input + token * columns;
    float accumulator = 0.0F;
    for (int64_t code_index = threadIdx.x;
            code_index < codes_per_row;
            code_index += blockDim.x) {
        const uint8_t code = codes[row * codes_per_row + code_index];
        const float lattice[4] = {
            static_cast<float>((code >> 6) & 3U) - 1.5F,
            static_cast<float>((code >> 4) & 3U) - 1.5F,
            static_cast<float>((code >> 2) & 3U) - 1.5F,
            static_cast<float>(code & 3U) - 1.5F,
        };
        const int64_t input_offset = code_index * 4;
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            float weight = group_bias[component];
#pragma unroll
            for (int lattice_index = 0; lattice_index < 4; ++lattice_index) {
                weight = fmaf(
                    lattice[lattice_index],
                    group_weight[component * 4 + lattice_index],
                    weight);
            }
            accumulator = fmaf(
                weight,
                token_input[input_offset + component],
                accumulator);
        }
    }

    accumulator = warp_sum(accumulator);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_sums[warp] = accumulator;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum =
            lane < matvec_threads / 32 ? warp_sums[lane] : 0.0F;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            output[token * rows + row] = block_sum;
        }
    }
}

__global__ void transform_output(
        const float * transformed,
        const int8_t * signs,
        float * output,
        int64_t rows,
        int64_t blocks_per_token) {
    __shared__ float values[transform_block];
    const int64_t token = blockIdx.x / blocks_per_token;
    const int64_t token_block = blockIdx.x % blocks_per_token;
    const int64_t block_start = token_block * transform_block;
    const int lane = threadIdx.x;
    const int64_t index = token * rows + block_start + lane;
    values[lane] = transformed[index];
    __syncthreads();

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
    output[index] = values[lane] * transform_scale *
        static_cast<float>(signs[block_start + lane]);
}

} // namespace

void ggml_cuda_op_btl3_avq2(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_is_btl3_avq2(dst));
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * affine = dst->src[1];
    const ggml_tensor * bias = dst->src[2];
    const ggml_tensor * input_signs = dst->src[3];
    const ggml_tensor * output_signs = dst->src[4];
    const ggml_tensor * input = dst->src[5];
    const int64_t rows = dst->ne[0];
    const int64_t columns = input->ne[0];
    const int64_t groups = bias->ne[1];
    const int64_t tokens = input->ne[1] * input->ne[2] * input->ne[3];
    const int64_t input_blocks = columns / transform_block;
    const int64_t output_blocks = rows / transform_block;

    ggml_cuda_pool_alloc<float> transformed_input(
        ctx.pool(), tokens * columns);
    ggml_cuda_pool_alloc<float> transformed_output(
        ctx.pool(), tokens * rows);
    cudaStream_t stream = ctx.stream();

    transform_input<<<input_blocks * tokens, transform_block, 0, stream>>>(
        static_cast<const float *>(input->data),
        static_cast<const int8_t *>(input_signs->data),
        transformed_input.get(),
        columns,
        input_blocks);
    decode_mul_mat<<<rows * tokens, matvec_threads, 0, stream>>>(
        static_cast<const uint8_t *>(codes->data),
        static_cast<const float *>(affine->data),
        static_cast<const float *>(bias->data),
        transformed_input.get(),
        transformed_output.get(),
        rows,
        columns,
        rows / groups);
    transform_output<<<output_blocks * tokens,
            transform_block, 0, stream>>>(
        transformed_output.get(),
        static_cast<const int8_t *>(output_signs->data),
        static_cast<float *>(dst->data),
        rows,
        output_blocks);
}

#else

void ggml_cuda_op_btl3_avq2(
        ggml_backend_cuda_context &,
        ggml_tensor *) {
    GGML_ABORT("BTL3 AVQ2 is only implemented for CUDA");
}

#endif
