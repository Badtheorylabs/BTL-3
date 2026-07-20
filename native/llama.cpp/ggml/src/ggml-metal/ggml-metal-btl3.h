#pragma once

#include "ggml-impl.h"

#include <stdint.h>
#include <string.h>

#define GGML_METAL_BTL3_AVQ2_MAGIC UINT64_C(0x42544c3341565132)
#define GGML_METAL_BTL3_INT4_MAGIC UINT64_C(0x42544c33494e5434)
#define GGML_METAL_BTL3_VOCAB_ROWS_MAGIC UINT64_C(0x42544c3356475257)
#define GGML_METAL_BTL3_VOCAB_HEAD_MAGIC UINT64_C(0x42544c3356484544)
#define GGML_METAL_BTL3_PACKED_PROBE_MAGIC UINT64_C(0x42544c3350524f42)
#define GGML_METAL_BTL3_METAL_PROBE_MAGIC UINT64_C(0x42544c334d505242)

static inline uintptr_t ggml_metal_btl3_custom_magic(
        const struct ggml_tensor * op) {
    if (op == NULL || op->op != GGML_OP_CUSTOM) {
        return 0;
    }
    struct ggml_custom_op_params params;
    memcpy(&params, op->op_params, sizeof(params));
    return (uintptr_t) params.userdata;
}

static inline bool ggml_metal_is_btl3_packed_probe(
        const struct ggml_tensor * op) {
    const uintptr_t magic = ggml_metal_btl3_custom_magic(op);
    if (magic != GGML_METAL_BTL3_PACKED_PROBE_MAGIC &&
        magic != GGML_METAL_BTL3_METAL_PROBE_MAGIC) {
        return false;
    }
    const struct ggml_tensor * packed = op->src[0];
    if (packed == NULL || op->src[1] != NULL ||
        !ggml_is_contiguous(packed)) {
        return false;
    }
    return packed->type == GGML_TYPE_I8 ||
        packed->type == GGML_TYPE_I16 ||
        packed->type == GGML_TYPE_I32 ||
        packed->type == GGML_TYPE_F16 ||
        packed->type == GGML_TYPE_F32;
}

static inline bool ggml_metal_is_btl3_avq2(const struct ggml_tensor * op) {
    if (ggml_metal_btl3_custom_magic(op) !=
        GGML_METAL_BTL3_AVQ2_MAGIC) {
        return false;
    }
    const struct ggml_tensor * codes = op->src[0];
    const struct ggml_tensor * affine = op->src[1];
    const struct ggml_tensor * bias = op->src[2];
    const struct ggml_tensor * input_signs = op->src[3];
    const struct ggml_tensor * output_signs = op->src[4];
    const struct ggml_tensor * input = op->src[5];
    if (codes == NULL || affine == NULL || bias == NULL ||
        input_signs == NULL || output_signs == NULL || input == NULL) {
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
        ggml_is_contiguous(codes) && ggml_is_contiguous(affine) &&
        ggml_is_contiguous(bias) && ggml_is_contiguous(input_signs) &&
        ggml_is_contiguous(output_signs) && ggml_is_contiguous(input);
}

static inline bool ggml_metal_is_btl3_int4(const struct ggml_tensor * op) {
    if (ggml_metal_btl3_custom_magic(op) !=
        GGML_METAL_BTL3_INT4_MAGIC) {
        return false;
    }
    const struct ggml_tensor * codes = op->src[0];
    const struct ggml_tensor * scales = op->src[1];
    const struct ggml_tensor * zeros = op->src[2];
    const struct ggml_tensor * input = op->src[3];
    if (codes == NULL || scales == NULL || zeros == NULL || input == NULL) {
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
        ggml_is_contiguous(codes) && ggml_is_contiguous(scales) &&
        ggml_is_contiguous(zeros) && ggml_is_contiguous(input);
}

static inline bool ggml_metal_is_btl3_vocab_base(
        const struct ggml_tensor * op,
        uintptr_t magic) {
    if (ggml_metal_btl3_custom_magic(op) != magic) {
        return false;
    }
    const struct ggml_tensor * codes = op->src[0];
    const struct ggml_tensor * affine = op->src[1];
    const struct ggml_tensor * bias = op->src[2];
    const struct ggml_tensor * signs = op->src[3];
    if (codes == NULL || affine == NULL || bias == NULL || signs == NULL) {
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

static inline bool ggml_metal_is_btl3_vocab_rows(
        const struct ggml_tensor * op) {
    if (!ggml_metal_is_btl3_vocab_base(
            op, GGML_METAL_BTL3_VOCAB_ROWS_MAGIC)) {
        return false;
    }
    const struct ggml_tensor * row_ids = op->src[4];
    const struct ggml_tensor * upper = op->src[5];
    const struct ggml_tensor * scales = op->src[6];
    const struct ggml_tensor * tokens = op->src[7];
    const int64_t columns = op->src[3]->ne[0];
    return row_ids != NULL && upper != NULL &&
        scales != NULL && tokens != NULL &&
        row_ids->type == GGML_TYPE_I32 &&
        upper->type == GGML_TYPE_I8 &&
        scales->type == GGML_TYPE_F16 &&
        tokens->type == GGML_TYPE_I32 &&
        upper->ne[0] == columns * 3 / 4 &&
        upper->ne[1] == row_ids->ne[0] &&
        scales->ne[0] == row_ids->ne[0] &&
        ggml_is_contiguous(row_ids) && ggml_is_contiguous(upper) &&
        ggml_is_contiguous(scales) && ggml_is_contiguous(tokens);
}

static inline bool ggml_metal_is_btl3_vocab_head(
        const struct ggml_tensor * op) {
    const struct ggml_tensor * input = op->src[4];
    return ggml_metal_is_btl3_vocab_base(
            op, GGML_METAL_BTL3_VOCAB_HEAD_MAGIC) &&
        input != NULL && input->type == GGML_TYPE_F32 &&
        input->ne[0] == op->src[3]->ne[0] &&
        ggml_is_contiguous(input);
}
