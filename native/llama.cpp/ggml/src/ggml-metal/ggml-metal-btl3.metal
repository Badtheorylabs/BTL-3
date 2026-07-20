struct ggml_metal_kargs_btl3_avq2 {
    int rows;
    int columns;
    int groups;
    int tokens;
};

kernel void kernel_btl3_avq2(
        constant ggml_metal_kargs_btl3_avq2 & args,
        device const uchar * codes,
        device const float * affine,
        device const float * bias,
        device const char * input_signs,
        device const char * output_signs,
        device const float * input,
        device float * output,
        threadgroup float * scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    constexpr float inverse_sqrt_128 = 0.08838834764831845f;
    const ushort tid = tpitg.x;
    threadgroup float * ping = scratch;
    threadgroup float * pong = scratch + 128;
    threadgroup float4 * codebook =
        reinterpret_cast<threadgroup float4 *>(scratch + 256);

    const int row_base = int(tgpig.x) * 128;
    const int token = int(tgpig.y);
    const int codebook_rows = args.rows / args.groups;
    const int group = row_base / codebook_rows;
    const device float * group_affine = affine + group * 16;
    const device float * group_bias = bias + group * 4;

    for (int code = tid; code < 256; code += 128) {
        const float4 lattice = float4(
            float((code >> 6) & 3) - 1.5f,
            float((code >> 4) & 3) - 1.5f,
            float((code >> 2) & 3) - 1.5f,
            float(code & 3) - 1.5f);
        codebook[code] = float4(
            group_bias[0] + dot(lattice, *((device const float4 *) group_affine)),
            group_bias[1] + dot(lattice, *((device const float4 *) (group_affine + 4))),
            group_bias[2] + dot(lattice, *((device const float4 *) (group_affine + 8))),
            group_bias[3] + dot(lattice, *((device const float4 *) (group_affine + 12))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int row = row_base + int(tid);
    const int codes_per_row = args.columns / 4;
    float accumulator = 0.0f;
    for (int block = 0; block < args.columns; block += 128) {
        const int column = block + int(tid);
        ping[tid] = input[token * args.columns + column] *
            float(input_signs[column]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int stride = 1; stride < 128; stride *= 2) {
            const int partner = int(tid) ^ stride;
            const float current = ping[tid];
            const float other = ping[partner];
            pong[tid] = (int(tid) & stride) != 0
                ? other - current
                : current + other;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float * swap = ping;
            ping = pong;
            pong = swap;
        }
        for (int code_index = 0; code_index < 32; ++code_index) {
            const uchar code =
                codes[row * codes_per_row + block / 4 + code_index];
            const float4 values = float4(
                ping[code_index * 4],
                ping[code_index * 4 + 1],
                ping[code_index * 4 + 2],
                ping[code_index * 4 + 3]);
            accumulator +=
                dot(codebook[code], values) * inverse_sqrt_128;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    ping[tid] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 1; stride < 128; stride *= 2) {
        const int partner = int(tid) ^ stride;
        const float current = ping[tid];
        const float other = ping[partner];
        pong[tid] = (int(tid) & stride) != 0
            ? other - current
            : current + other;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * swap = ping;
        ping = pong;
        pong = swap;
    }
    output[token * args.rows + row] =
        ping[tid] * inverse_sqrt_128 * float(output_signs[row]);
}

struct ggml_metal_kargs_btl3_int4 {
    int rows;
    int columns;
    int tokens;
};

kernel void kernel_btl3_int4(
        constant ggml_metal_kargs_btl3_int4 & args,
        device const uchar * codes,
        device const half * scales,
        device const uchar * zeros,
        device const float * input,
        device float * output,
        threadgroup float * partial,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    const ushort tid = tpitg.x;
    const int row = int(tgpig.x);
    const int token = int(tgpig.y);
    const int groups_per_row = args.columns / 128;
    float accumulator = 0.0f;
    for (int group = 0; group < groups_per_row; ++group) {
        const int column = group * 128 + int(tid);
        const uchar packed =
            codes[row * (args.columns / 2) + column / 2];
        const uchar code =
            (column & 1) == 0 ? packed & 15 : packed >> 4;
        const int group_index = row * groups_per_row + group;
        const float weight =
            (float(code) - float(zeros[group_index])) *
            float(scales[group_index]);
        accumulator +=
            weight * input[token * args.columns + column];
    }
    partial[tid] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 64; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        output[token * args.rows + row] = partial[0];
    }
}

struct ggml_metal_kargs_btl3_vocab_head {
    int vocabulary;
    int columns;
    int groups;
    int tokens;
};

kernel void kernel_btl3_vocab_head(
        constant ggml_metal_kargs_btl3_vocab_head & args,
        device const uchar * codes,
        device const float * affine,
        device const float * bias,
        device const char * input_signs,
        device const float * input,
        device float * output,
        threadgroup float * scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    constexpr float inverse_sqrt_128 = 0.08838834764831845f;
    const ushort tid = tpitg.x;
    threadgroup float * ping = scratch;
    threadgroup float * pong = scratch + 128;
    threadgroup float4 * codebook =
        reinterpret_cast<threadgroup float4 *>(scratch + 256);
    const int row = int(tgpig.x) * 128 + int(tid);
    const int token = int(tgpig.y);
    const int codebook_rows = args.vocabulary / args.groups;
    const int group = row / codebook_rows;
    const device float * group_affine = affine + group * 16;
    const device float * group_bias = bias + group * 4;

    for (int code = tid; code < 256; code += 128) {
        const float4 lattice = float4(
            float((code >> 6) & 3) - 1.5f,
            float((code >> 4) & 3) - 1.5f,
            float((code >> 2) & 3) - 1.5f,
            float(code & 3) - 1.5f);
        codebook[code] = float4(
            group_bias[0] + dot(lattice, *((device const float4 *) group_affine)),
            group_bias[1] + dot(lattice, *((device const float4 *) (group_affine + 4))),
            group_bias[2] + dot(lattice, *((device const float4 *) (group_affine + 8))),
            group_bias[3] + dot(lattice, *((device const float4 *) (group_affine + 12))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int codes_per_row = args.columns / 4;
    float accumulator = 0.0f;
    for (int block = 0; block < args.columns; block += 128) {
        const int column = block + int(tid);
        ping[tid] = input[token * args.columns + column] *
            float(input_signs[column]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int stride = 1; stride < 128; stride *= 2) {
            const int partner = int(tid) ^ stride;
            const float current = ping[tid];
            const float other = ping[partner];
            pong[tid] = (int(tid) & stride) != 0
                ? other - current
                : current + other;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float * swap = ping;
            ping = pong;
            pong = swap;
        }
        for (int code_index = 0; code_index < 32; ++code_index) {
            const uchar code =
                codes[row * codes_per_row + block / 4 + code_index];
            const float4 values = float4(
                ping[code_index * 4],
                ping[code_index * 4 + 1],
                ping[code_index * 4 + 2],
                ping[code_index * 4 + 3]);
            accumulator +=
                dot(codebook[code], values) * inverse_sqrt_128;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[token * args.vocabulary + row] = accumulator;
}

struct ggml_metal_kargs_btl3_vocab_rows {
    int vocabulary;
    int columns;
    int groups;
    int rescue_count;
    int tokens;
};

kernel void kernel_btl3_vocab_rows(
        constant ggml_metal_kargs_btl3_vocab_rows & args,
        device const uchar * codes,
        device const float * affine,
        device const float * bias,
        device const char * input_signs,
        device const int * rescued_row_ids,
        device const uchar * rescued_upper_six,
        device const half * rescued_scales,
        device const int * token_ids,
        device float * output,
        threadgroup float * scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]]) {
    constexpr float inverse_sqrt_128 = 0.08838834764831845f;
    const ushort tid = tpitg.x;
    const int column = int(tgpig.x) * 128 + int(tid);
    const int token_index = int(tgpig.y);
    const int token = token_ids[token_index];
    if (token < 0 || token >= args.vocabulary) {
        output[token_index * args.columns + column] = 0.0f;
        return;
    }
    threadgroup int * rescue_slot =
        reinterpret_cast<threadgroup int *>(scratch + 1280);
    if (tid == 0) {
        int low = 0;
        int high = args.rescue_count;
        while (low < high) {
            const int middle = low + (high - low) / 2;
            if (rescued_row_ids[middle] < token) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        *rescue_slot =
            low < args.rescue_count && rescued_row_ids[low] == token
                ? low
                : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int rescue = *rescue_slot;
    if (rescue >= 0) {
        const uchar low_byte =
            codes[token * (args.columns / 4) + column / 4];
        const uchar low_two =
            (low_byte >> ((column & 3) * 2)) & 3;
        const int upper_index = column;
        const int upper_group = upper_index / 4;
        const int upper_lane = upper_index & 3;
        const int upper_stride = args.columns * 3 / 4;
        const device uchar * upper =
            rescued_upper_six + rescue * upper_stride + upper_group * 3;
        const uint packed_upper =
            uint(upper[0]) | (uint(upper[1]) << 8) | (uint(upper[2]) << 16);
        const uchar high_six =
            uchar((packed_upper >> (upper_lane * 6)) & 63);
        const uchar packed = low_two | (high_six << 2);
        output[token_index * args.columns + column] =
            (float(packed) - 128.0f) * float(rescued_scales[rescue]);
        return;
    }

    threadgroup float * ping = scratch;
    threadgroup float * pong = scratch + 128;
    threadgroup float4 * codebook =
        reinterpret_cast<threadgroup float4 *>(scratch + 256);
    const int codebook_rows = args.vocabulary / args.groups;
    const int group = token / codebook_rows;
    const device float * group_affine = affine + group * 16;
    const device float * group_bias = bias + group * 4;
    for (int code = tid; code < 256; code += 128) {
        const float4 lattice = float4(
            float((code >> 6) & 3) - 1.5f,
            float((code >> 4) & 3) - 1.5f,
            float((code >> 2) & 3) - 1.5f,
            float(code & 3) - 1.5f);
        codebook[code] = float4(
            group_bias[0] + dot(lattice, *((device const float4 *) group_affine)),
            group_bias[1] + dot(lattice, *((device const float4 *) (group_affine + 4))),
            group_bias[2] + dot(lattice, *((device const float4 *) (group_affine + 8))),
            group_bias[3] + dot(lattice, *((device const float4 *) (group_affine + 12))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uchar code =
        codes[token * (args.columns / 4) + column / 4];
    ping[tid] = codebook[code][column & 3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = 1; stride < 128; stride *= 2) {
        const int partner = int(tid) ^ stride;
        const float current = ping[tid];
        const float other = ping[partner];
        pong[tid] = (int(tid) & stride) != 0
            ? other - current
            : current + other;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * swap = ping;
        ping = pong;
        pong = swap;
    }
    output[token_index * args.columns + column] =
        ping[tid] * inverse_sqrt_128 * float(input_signs[column]);
}
