#pragma once

#include "btl3-avq-marker.h"
#include "common.cuh"

void ggml_cuda_op_btl3_avq2(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst);
