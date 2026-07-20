#include "btl3-int4.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {

constexpr int group_size = 128;

__global__ void int4_mul_mat(
        const uint8_t * codes,
        const half * scales,
        const uint8_t * zeros,
        const float * input,
        float * output,
        int64_t rows,
        int64_t columns,
        int64_t groups,
        int64_t tokens) {
    __shared__ float partial[group_size];
    const int lane = threadIdx.x;
    const int64_t row = blockIdx.x % rows;
    const int64_t token = blockIdx.x / rows;
    if (token >= tokens) {
        return;
    }

    float accumulator = 0.0F;
    for (int64_t group = 0; group < groups; ++group) {
        const int64_t column = group * group_size + lane;
        const uint8_t packed =
            codes[row * (columns / 2) + column / 2];
        const uint8_t code =
            (column & 1) == 0 ? packed & 0x0FU : packed >> 4;
        const int64_t parameter = row * groups + group;
        const float weight =
            (static_cast<float>(code) -
             static_cast<float>(zeros[parameter])) *
            __half2float(scales[parameter]);
        accumulator = fmaf(
            weight,
            input[token * columns + column],
            accumulator);
    }

    partial[lane] = accumulator;
    __syncthreads();
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[token * rows + row] = partial[0];
    }
}

} // namespace

void ggml_cuda_op_btl3_int4(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_is_btl3_int4(dst));
    const ggml_tensor * codes = dst->src[0];
    const ggml_tensor * scales = dst->src[1];
    const ggml_tensor * zeros = dst->src[2];
    const ggml_tensor * input = dst->src[3];
    const int64_t rows = dst->ne[0];
    const int64_t columns = input->ne[0];
    const int64_t groups = columns / group_size;
    const int64_t tokens = input->ne[1] * input->ne[2] * input->ne[3];

    int4_mul_mat<<<rows * tokens, group_size, 0, ctx.stream()>>>(
        static_cast<const uint8_t *>(codes->data),
        static_cast<const half *>(scales->data),
        static_cast<const uint8_t *>(zeros->data),
        static_cast<const float *>(input->data),
        static_cast<float *>(dst->data),
        rows,
        columns,
        groups,
        tokens);
}

#else

void ggml_cuda_op_btl3_int4(
        ggml_backend_cuda_context &,
        ggml_tensor *) {
    GGML_ABORT("BTL3 INT4 is only implemented for CUDA");
}

#endif
