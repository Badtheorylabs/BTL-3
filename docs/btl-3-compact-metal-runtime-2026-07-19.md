# BTL-3 Compact Metal runtime validation

## Verified result

The native compact GGUF executes end to end on Apple Metal without rebuilding
dense weights. The tested artifact was:

- `artifacts/release/BTL-3-Compact-AVQ2.gguf`
- SHA-256:
  `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c`
- File size reported by the loader: 7.81 GiB

The Metal backend now has native implementations for:

- AVQ2 decoder projections;
- affine INT4 projections;
- compressed vocabulary output;
- compressed embedding lookup, including rescued rows.

Packed tensor placement uses two markers. Repeating AVQ2 tensors use the
cross-backend marker supported by Metal and CUDA. INT4, rescued vocabulary,
and non-repeating vocabulary AVQ tensors use a Metal-only marker. This keeps
CUDA from accepting storage for custom operations it cannot execute.

## Correctness gates

All local parity gates passed:

- AVQ2 4096 x 4096 x 4: max relative error `3.66604e-05`;
- INT4: max relative error `2.15352e-06`;
- vocabulary head: max relative error `1.43051e-06`;
- embedding rows: max relative error `5.96046e-08`.

The full GGUF also passed the model-load test and one OpenAI-compatible server
generation smoke.

## M2 speed

Test host: base Apple M2 with 16 GB unified memory. Server settings used one
slot, 128-token context, all layers requested on GPU, prompt cache disabled,
`--no-mmap`, and `--no-warmup`.

| Measurement | Before placement fix | After placement fix |
|---|---:|---:|
| Prompt processing | 0.609 tok/s | 2.297 tok/s |
| Token generation | 0.178 tok/s | 2.483 tok/s |

Generation improved by 13.9x. The post-fix response completed without a
backend error and emitted coherent reasoning tokens.

Isolated warm Metal measurements:

- AVQ2 4096 x 4096 x 4: 2.49 ms, 42.8x over the scalar CPU reference;
- full 248,320 x 5,120 vocabulary head: 22.15 ms;
- weighted representative AVQ2 decoder suite: about 735.6 ms;
- weighted representative INT4 decoder suite: about 70.4 ms.

## Boundaries

The M2 result is a functional first native speed result, not the final
performance ceiling. AVQ2 remains the dominant custom operator and its current
fused kernel repeats input transforms across output-row blocks. A future
three-stage implementation can compute the input transform once, run a
coalesced decoder matmul, and transform the output in place.

CUDA currently accelerates repeating AVQ2 decoder projections. It does not yet
implement the INT4 or compressed vocabulary custom operations, so no
end-to-end CUDA throughput claim should be made from the standalone CUDA
projection benchmark.
