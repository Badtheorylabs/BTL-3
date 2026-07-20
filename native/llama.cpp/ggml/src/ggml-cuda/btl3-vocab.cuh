#pragma once

#include "common.cuh"
#include "btl3-avq-marker.h"

void ggml_cuda_op_btl3_vocab_rows(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

void ggml_cuda_op_btl3_vocab_head(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
