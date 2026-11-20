<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# npu_emu

`libnpu_emu.so` reproduces NPU kernels bit for bit on an AMD GPU. It is a
standalone HIP library with a C ABI and does not depend on the rest of hip-ep.

Tensors use the NPU's element types: f32, f16, bf16, fp8 (E4M3FN), 8 / 16 / 32 /
64-bit integers, and the block floating point formats bfp16 and mx6. Every
entry point is asynchronous on a `hipStream_t` and returns a `hipError_t`
value, or -1 for unsupported arguments.

## Layout

| Folder | Kernels |
|---|---|
| `common/` | shared types, rounding, exact multiply-add, float and block-float codecs, MMAC and hardware function models, test helpers |
| `movement/` | data movement and selection |
| `convert/` | cast, convert, quantize, dequantize, requantize |
| `arithmetic/` | elementwise arithmetic |
| `nonlinear/` | nonlinear and transcendental functions |
| `reduction/` | reduction and normalization |
| `interpolation/` | grid sample, resize, RoPE |
| `matmul/` | matrix multiply, convolution, LSTM |
| `attention/` | flash attention |

A folder holds its C headers (`*.h`), HIP sources (`*.hip`), numpy reference
models (`golden_*.py`) and tests. Kernel folders depend only on `common/`
(`matmul/conv_instancenorm` also uses the instance norm of `reduction/`).

## Build and test

    make HIPCC=<path to hipcc> [ARCH=gfx1151]
    make test [PYTHON=python3] [K=<pytest -k expression>]

`test_golden_*.py` compare every entry point with its numpy reference.
`test_fixture_*.py` compare with NPU reference outputs and are skipped unless a
source is given:

- `NPU_EMU_MODEL_REPO`: a checkout of the model repository; its reference model
  generates each case on first use (interpreter `NPU_EMU_MODEL_PYTHON`; cases
  are kept in `NPU_EMU_FIXTURE_CACHE` if set, else in a temporary directory
  removed at exit).
- `NPU_EMU_SIM_ROOT`: recorded simulator outputs (all folders except `matmul/`).

`NPU_EMU_FIXTURE_LIMIT=<n>` runs at most n cases per directory.
