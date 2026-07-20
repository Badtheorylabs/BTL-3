#pragma once

#include "../ggml-impl.h"
#include "ggml.h"

#include <cstdint>
#include <cstring>

constexpr std::uintptr_t GGML_CUDA_BTL3_AVQ2_MAGIC =
    UINT64_C(0x42544c3341565132);
constexpr std::uintptr_t GGML_CUDA_BTL3_INT4_MAGIC =
    UINT64_C(0x42544c33494e5434);
constexpr std::uintptr_t GGML_CUDA_BTL3_VOCAB_ROWS_MAGIC =
    UINT64_C(0x42544c3356475257);
constexpr std::uintptr_t GGML_CUDA_BTL3_VOCAB_HEAD_MAGIC =
    UINT64_C(0x42544c3356484544);
constexpr std::uintptr_t GGML_CUDA_BTL3_AVQ2_PROBE_MAGIC =
    UINT64_C(0x42544c3343415651);
constexpr std::uintptr_t GGML_CUDA_BTL3_INT4_PROBE_MAGIC =
    UINT64_C(0x42544c3343494e54);
constexpr std::uintptr_t GGML_CUDA_BTL3_VOCAB_ROWS_PROBE_MAGIC =
    UINT64_C(0x42544c3343565257);
constexpr std::uintptr_t GGML_CUDA_BTL3_VOCAB_HEAD_PROBE_MAGIC =
    UINT64_C(0x42544c3343564844);

static inline std::uintptr_t ggml_cuda_custom_magic(
        const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM) {
        return 0;
    }
    ggml_custom_op_params params;
    std::memcpy(&params, op->op_params, sizeof(params));
    return reinterpret_cast<std::uintptr_t>(params.userdata);
}

static inline bool ggml_cuda_is_btl3_probe(
        const ggml_tensor * op,
        std::uintptr_t magic) {
    if (ggml_cuda_custom_magic(op) != magic ||
        op->src[0] == nullptr || op->src[1] != nullptr) {
        return false;
    }
    return ggml_is_contiguous(op->src[0]);
}

static inline bool ggml_cuda_is_btl3_avq2_probe(const ggml_tensor * op) {
    return ggml_cuda_is_btl3_probe(
            op, GGML_CUDA_BTL3_AVQ2_PROBE_MAGIC) &&
        (op->src[0]->type == GGML_TYPE_I8 ||
         op->src[0]->type == GGML_TYPE_F32);
}

static inline bool ggml_cuda_is_btl3_int4_probe(const ggml_tensor * op) {
    return ggml_cuda_is_btl3_probe(
            op, GGML_CUDA_BTL3_INT4_PROBE_MAGIC) &&
        (op->src[0]->type == GGML_TYPE_I8 ||
         op->src[0]->type == GGML_TYPE_F16);
}

static inline bool ggml_cuda_is_btl3_vocab_rows_probe(
        const ggml_tensor * op) {
    return ggml_cuda_is_btl3_probe(
            op, GGML_CUDA_BTL3_VOCAB_ROWS_PROBE_MAGIC) &&
        (op->src[0]->type == GGML_TYPE_I8 ||
         op->src[0]->type == GGML_TYPE_I32 ||
         op->src[0]->type == GGML_TYPE_F16 ||
         op->src[0]->type == GGML_TYPE_F32);
}

static inline bool ggml_cuda_is_btl3_vocab_head_probe(
        const ggml_tensor * op) {
    return ggml_cuda_is_btl3_probe(
            op, GGML_CUDA_BTL3_VOCAB_HEAD_PROBE_MAGIC) &&
        (op->src[0]->type == GGML_TYPE_I8 ||
         op->src[0]->type == GGML_TYPE_F32);
}

static inline bool ggml_cuda_is_btl3_int4(const ggml_tensor * op) {
    if (ggml_cuda_custom_magic(op) != GGML_CUDA_BTL3_INT4_MAGIC) {
        return false;
    }
    const ggml_tensor * codes = op->src[0];
    const ggml_tensor * scales = op->src[1];
    const ggml_tensor * zeros = op->src[2];
    const ggml_tensor * input = op->src[3];
    if (codes == nullptr || scales == nullptr ||
        zeros == nullptr || input == nullptr) {
        return false;
    }
    const int64_t columns = input->ne[0];
    const int64_t rows = codes->ne[1];
    return op->type == GGML_TYPE_F32 &&
        codes->type == GGML_TYPE_I8 &&
        scales->type == GGML_TYPE_F16 &&
        zeros->type == GGML_TYPE_I8 &&
        input->type == GGML_TYPE_F32 &&
        columns > 0 && rows > 0 && columns % 128 == 0 &&
        codes->ne[0] == columns / 2 &&
        scales->ne[0] == columns / 128 && scales->ne[1] == rows &&
        zeros->ne[0] == columns / 128 && zeros->ne[1] == rows &&
        op->ne[0] == rows &&
        op->ne[1] == input->ne[1] &&
        op->ne[2] == input->ne[2] &&
        op->ne[3] == input->ne[3] &&
        ggml_is_contiguous(codes) && ggml_is_contiguous(scales) &&
        ggml_is_contiguous(zeros) && ggml_is_contiguous(input) &&
        ggml_is_contiguous(op);
}

static inline bool ggml_cuda_is_btl3_vocab_base(
        const ggml_tensor * op,
        std::uintptr_t magic) {
    if (ggml_cuda_custom_magic(op) != magic) {
        return false;
    }
    const ggml_tensor * codes = op->src[0];
    const ggml_tensor * affine = op->src[1];
    const ggml_tensor * bias = op->src[2];
    const ggml_tensor * signs = op->src[3];
    if (codes == nullptr || affine == nullptr ||
        bias == nullptr || signs == nullptr) {
        return false;
    }
    const int64_t columns = signs->ne[0];
    const int64_t vocabulary = codes->ne[1];
    const int64_t groups = bias->ne[1];
    return op->type == GGML_TYPE_F32 &&
        codes->type == GGML_TYPE_I8 &&
        affine->type == GGML_TYPE_F32 &&
        bias->type == GGML_TYPE_F32 &&
        signs->type == GGML_TYPE_I8 &&
        columns > 0 && vocabulary > 0 && groups > 0 &&
        columns % 128 == 0 && vocabulary % 128 == 0 &&
        vocabulary % groups == 0 &&
        (vocabulary / groups) % 128 == 0 &&
        codes->ne[0] == columns / 4 &&
        affine->ne[0] == 4 && affine->ne[1] == 4 &&
        affine->ne[2] == groups && bias->ne[0] == 4 &&
        ggml_is_contiguous(codes) && ggml_is_contiguous(affine) &&
        ggml_is_contiguous(bias) && ggml_is_contiguous(signs);
}

static inline bool ggml_cuda_is_btl3_vocab_rows(const ggml_tensor * op) {
    if (!ggml_cuda_is_btl3_vocab_base(
            op, GGML_CUDA_BTL3_VOCAB_ROWS_MAGIC)) {
        return false;
    }
    const ggml_tensor * row_ids = op->src[4];
    const ggml_tensor * upper = op->src[5];
    const ggml_tensor * scales = op->src[6];
    const ggml_tensor * tokens = op->src[7];
    const int64_t columns = op->src[3]->ne[0];
    return row_ids != nullptr && upper != nullptr &&
        scales != nullptr && tokens != nullptr &&
        row_ids->type == GGML_TYPE_I32 &&
        upper->type == GGML_TYPE_I8 &&
        scales->type == GGML_TYPE_F16 &&
        tokens->type == GGML_TYPE_I32 &&
        upper->ne[0] == columns * 3 / 4 &&
        upper->ne[1] == row_ids->ne[0] &&
        scales->ne[0] == row_ids->ne[0] &&
        op->ne[0] == columns &&
        op->ne[1] * op->ne[2] * op->ne[3] ==
            tokens->ne[0] * tokens->ne[1] * tokens->ne[2] * tokens->ne[3] &&
        ggml_is_contiguous(row_ids) && ggml_is_contiguous(upper) &&
        ggml_is_contiguous(scales) && ggml_is_contiguous(tokens) &&
        ggml_is_contiguous(op);
}

static inline bool ggml_cuda_is_btl3_vocab_head(const ggml_tensor * op) {
    const ggml_tensor * input = op->src[4];
    return ggml_cuda_is_btl3_vocab_base(
            op, GGML_CUDA_BTL3_VOCAB_HEAD_MAGIC) &&
        input != nullptr && input->type == GGML_TYPE_F32 &&
        input->ne[0] == op->src[3]->ne[0] &&
        op->ne[0] == op->src[0]->ne[1] &&
        op->ne[1] == input->ne[1] &&
        op->ne[2] == input->ne[2] &&
        op->ne[3] == input->ne[3] &&
        ggml_is_contiguous(input) && ggml_is_contiguous(op);
}

static inline bool ggml_cuda_is_btl3_avq2(const ggml_tensor * op) {
    if (ggml_cuda_custom_magic(op) != GGML_CUDA_BTL3_AVQ2_MAGIC) {
        return false;
    }
    const ggml_tensor * codes = op->src[0];
    const ggml_tensor * affine = op->src[1];
    const ggml_tensor * bias = op->src[2];
    const ggml_tensor * input_signs = op->src[3];
    const ggml_tensor * output_signs = op->src[4];
    const ggml_tensor * input = op->src[5];
    if (codes == nullptr || affine == nullptr || bias == nullptr ||
        input_signs == nullptr || output_signs == nullptr ||
        input == nullptr) {
        return false;
    }
    const int64_t columns = input->ne[0];
    const int64_t rows = output_signs->ne[0];
    const int64_t groups = bias->ne[1];
    const int64_t codebook_rows = groups > 0 ? rows / groups : 0;
    return op->type == GGML_TYPE_F32 &&
        codes->type == GGML_TYPE_I8 &&
        affine->type == GGML_TYPE_F32 &&
        bias->type == GGML_TYPE_F32 &&
        input_signs->type == GGML_TYPE_I8 &&
        output_signs->type == GGML_TYPE_I8 &&
        input->type == GGML_TYPE_F32 &&
        columns > 0 && rows > 0 && groups > 0 &&
        columns % 128 == 0 && rows % 128 == 0 &&
        rows % groups == 0 && codebook_rows % 128 == 0 &&
        codes->ne[0] == columns / 4 && codes->ne[1] == rows &&
        affine->ne[0] == 4 && affine->ne[1] == 4 &&
        affine->ne[2] == groups && bias->ne[0] == 4 &&
        input_signs->ne[0] == columns &&
        op->ne[0] == rows &&
        op->ne[1] == input->ne[1] &&
        op->ne[2] == input->ne[2] &&
        op->ne[3] == input->ne[3] &&
        ggml_is_contiguous(codes) && ggml_is_contiguous(affine) &&
        ggml_is_contiguous(bias) && ggml_is_contiguous(input_signs) &&
        ggml_is_contiguous(output_signs) && ggml_is_contiguous(input) &&
        ggml_is_contiguous(op);
}
